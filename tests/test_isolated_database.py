import ctypes
import hashlib
import os
import multiprocessing as mp
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

import database_access as access
from isolated_database import IsolatedCipherConnection, DEFAULT_PENDING, _set_mapping


class Layout:
    """Model two boundary pages without allocating a 1-GiB test file."""
    def __init__(self, page=4096, shifted_zero=True, normal_data=True):
        self.page = page
        self.zero = shifted_zero
        self.data = normal_data

    def stat(self):
        return SimpleNamespace(st_size=0x40000000 + self.page + 1024)

    def open(self, _):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def seek(self, offset):
        self.offset = offset

    def read(self, page):
        shifted = 1024 + ((0x40000000 - 1024) // page) * page
        if self.offset == shifted:
            return bytes(page) if self.zero else b"x" * page
        return b"x" * page if self.data else bytes(page)


def test_layout_requires_offset_opaque_and_both_page_evidence():
    cfg = {"cipher_page_size": 4096}
    assert access._compat_layout(Layout(), 1024, False, cfg)
    assert not access._compat_layout(Layout(), 0, False, cfg)
    assert not access._compat_layout(Layout(), 1024, True, cfg)
    assert not access._compat_layout(Layout(shifted_zero=False), 1024, False, cfg)
    assert not access._compat_layout(Layout(normal_data=False), 1024, False, cfg)
    with pytest.raises(access.DatabaseError, match="已验证范围"):
        access._compat_layout(Layout(page=8192), 1024, False, {"cipher_page_size": 8192})


def test_mapping_refuses_nondefault_process_state(monkeypatch):
    import isolated_database
    monkeypatch.setattr(isolated_database.mp, 'current_process', lambda: SimpleNamespace(name='tei-nt-readonly'))
    class Control:
        def __call__(self, *_):
            return 123
    with pytest.raises(access.DatabaseError) as exc:
        _set_mapping(SimpleNamespace(sqlite3_test_control=Control()), 0x3ffffc00)
    assert exc.value.code == "compat_unsupported"


@pytest.mark.parametrize(('group', 'private'), [
    ({'40027', '40033', '40050', '40800'}, set()),
    ({'40027', '40033', '40050', '40800'}, {'40033'}),
    (set(), {'40030', '40033', '40050', '40800'}),
    ({'40001'}, {'40030', '40033', '40050', '40800'}),
])
def test_schema_accepts_usable_mode_without_optional_key_or_other_table(group, private):
    from isolated_database import _check_schema
    class Schema:
        def query(self, sql):
            if sql == 'PRAGMA page_size':
                return [{'page_size': '4096'}]
            columns = group if 'group_msg_table' in sql else private
            return [{'name': column} for column in columns]
    _check_schema(Schema())


@pytest.mark.parametrize(('group', 'private', 'page'), [
    ({'40030', '40033', '40050', '40800'}, set(), 4096),
    (set(), {'40027', '40033', '40050', '40800'}, 4096),
    ({'40027', '40033', '40050'}, {'40030', '40033', '40800'}, 4096),
    ({'40027', '40033', '40050', '40800'}, set(), 8192),
])
def test_schema_rejects_incomplete_or_wrong_mode_and_page(group, private, page):
    from isolated_database import _check_schema
    class Schema:
        def query(self, sql):
            if sql == 'PRAGMA page_size':
                return [{'page_size': str(page)}]
            return [{'name': column} for column in (group if 'group_msg_table' in sql else private)]
    with pytest.raises(access.DatabaseError) as exc:
        _check_schema(Schema())
    assert exc.value.code == 'compat_unsupported'


def test_standard_open_does_not_hash_unrelated_pages(tmp_path, monkeypatch):
    path = tmp_path / 'normal.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE indexed_read (id INTEGER PRIMARY KEY)')
        db.execute('INSERT INTO indexed_read VALUES (1)')
    def forbidden_hash(*_):
        raise AssertionError('standard opens must not do a full-file hash pass')
    monkeypatch.setattr(access.hashlib, 'sha256', forbidden_hash)
    with access.open_database(path) as (db, encrypted, receipt):
        assert db.query('SELECT id FROM indexed_read WHERE id = 1') == [{'id': 1}]
        assert not encrypted
        assert receipt['source_sha256'] is None and receipt['payload_sha256'] is None
        assert receipt['source_hash_status'] == 'not_computed_standard'


def test_prefix_copy_still_records_source_and_payload_hashes(tmp_path):
    payload = tmp_path / 'payload.db'
    with sqlite3.connect(payload) as db:
        db.execute('CREATE TABLE readable (id)')
    path = tmp_path / 'prefixed.db'
    body = payload.read_bytes()
    source = b'P' * 1024 + body
    path.write_bytes(source)
    with access.open_database(path, scratch_dir=tmp_path) as (_, _, receipt):
        assert receipt['source_sha256'] == hashlib.sha256(source).hexdigest()
        assert receipt['payload_sha256'] == hashlib.sha256(body).hexdigest()
        assert receipt['source_hash_status'] == 'computed_during_prefix_copy'
    assert path.read_bytes() == source
    assert not list(tmp_path.glob('tei-payload-*'))


@pytest.fixture
def encrypted(tmp_path):
    dll = Path(os.environ.get("TEI_TEST_SQLCIPHER_DLL", "C:/msys64/mingw64/bin/libsqlcipher-0.dll"))
    if os.name != "nt" or not dll.is_file():
        pytest.skip("Requires the locally installed Windows SQLCipher; no download")
    directory = os.add_dll_directory(str(dll.parent))
    lib = ctypes.CDLL(str(dll))
    lib.sqlite3_open_v2.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_void_p]
    lib.sqlite3_key.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.sqlite3_exec.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.sqlite3_close.argtypes = [ctypes.c_void_p]
    path = tmp_path / "synthetic encrypted.db"
    handle = ctypes.c_void_p()
    assert lib.sqlite3_open_v2(str(path).encode(), ctypes.byref(handle), 6, None) == 0
    key = b"synthetic-only"
    assert lib.sqlite3_key(handle, key, len(key)) == 0
    sql = ("PRAGMA cipher_page_size=4096; PRAGMA kdf_iter=4000;"
           "PRAGMA cipher_hmac_algorithm=HMAC_SHA1; PRAGMA cipher_kdf_algorithm=PBKDF2_HMAC_SHA512;"
           "CREATE TABLE group_msg_table([40027],[40033],[40050],[40800]);"
           "CREATE TABLE c2c_msg_table([40030],[40033],[40050],[40800]);"
           "INSERT INTO group_msg_table VALUES(99,1,1700000000,'synthetic');"
           "INSERT INTO group_msg_table VALUES(99,2,1700000001,'synthetic');"
           "INSERT INTO c2c_msg_table VALUES(9,1,1700000000,'synthetic');")
    assert lib.sqlite3_exec(handle, sql.encode(), None, None, None) == 0
    lib.sqlite3_close(handle)
    cfg = {"password": key.decode(), "sqlcipher_dll": str(dll), "cipher_page_size": 4096, "kdf_iter": 4000}
    probe = access.CipherConnection(path, cfg, lambda: False)
    try:
        versions = (probe.query('SELECT sqlite_version() AS v')[0]['v'],
                    probe.query('PRAGMA cipher_version')[0]['cipher_version'])
    finally:
        probe.close()
    from isolated_database import SUPPORTED
    if versions != SUPPORTED:
        directory.close()
        pytest.skip('Host SQLCipher is outside the explicitly supported compatibility versions')
    yield path, cfg, lib
    directory.close()


def test_real_worker_stream_repeat_cancel_and_parent_mapping(encrypted):
    path, cfg, lib = encrypted
    before = hashlib.sha256(path.read_bytes()).digest()
    assert lib.sqlite3_test_control(ctypes.c_int(11), ctypes.c_uint(0)) == DEFAULT_PENDING
    # Direct worker tests exercise real C/IPC/containment on small synthetic NT
    # databases. Production routing additionally requires the 1-GiB layout gate.
    for _ in range(2):
        c = IsolatedCipherConnection(path, cfg, lambda: False)
        assert c.query('SELECT count(*) AS n FROM group_msg_table') == [{"n": "2"}]
        assert c.query('SELECT [40030] AS peer FROM c2c_msg_table') == [{'peer': '9'}]
        assert len(list(c.iter_query('SELECT [40800] FROM group_msg_table'))) == 2
        c.close()
        c.close()
    cancel = [False]
    c = IsolatedCipherConnection(path, cfg, lambda: cancel[0])
    cancel[0] = True
    with pytest.raises(access.DatabaseError) as exc:
        c.query('SELECT count(*) AS n FROM group_msg_table')
    assert exc.value.code == 'cancelled' and c.closed
    assert lib.sqlite3_test_control(ctypes.c_int(11), ctypes.c_uint(0)) == DEFAULT_PENDING
    assert before == hashlib.sha256(path.read_bytes()).digest()
    with sqlite3.connect(':memory:') as db:
        assert db.execute('SELECT 1').fetchone() == (1,)


def test_worker_two_connections_early_stop_and_write_refusal(encrypted):
    path, cfg, _ = encrypted
    first = IsolatedCipherConnection(path, cfg, lambda: False)
    second = IsolatedCipherConnection(path, cfg, lambda: False)
    try:
        it = first.iter_query('SELECT [40800] FROM group_msg_table')
        next(it)
        assert second.query('SELECT count(*) AS n FROM group_msg_table') == [{"n": "2"}]
        it.close()
        assert first.closed
        with pytest.raises(access.DatabaseError):
            second.query("ATTACH 'must-not-create.db' AS forbidden")
        assert second.closed
    finally:
        first.close()
        second.close()


def test_worker_real_corruption_and_schema_rejection(encrypted):
    path, cfg, _ = encrypted
    data = bytearray(path.read_bytes())
    data[4096:8192] = bytes(4096)  # The actual group table root, not a guessed SQL failure.
    path.write_bytes(data)
    c = IsolatedCipherConnection(path, cfg, lambda: False)
    try:
        with pytest.raises(access.DatabaseError) as exc:
            c.query('SELECT count(*) AS n FROM group_msg_table')
        assert exc.value.code == 'corrupt'
    finally:
        c.close()


def test_wrong_password_is_not_compatibility(encrypted):
    path, cfg, _ = encrypted
    conn = access.CipherConnection(path, cfg, lambda: False)
    assert conn.query('SELECT count(*) AS n FROM sqlite_master')
    conn.close()
    # A wrong password must fail safely before any mapping change.
    with pytest.raises(access.DatabaseError) as exc:
        IsolatedCipherConnection(path, dict(cfg, password='wrong-synthetic'), lambda: False)
    assert exc.value.code == 'unlock'


def test_unrelated_schema_and_main_process_are_rejected():
    from isolated_database import _check_schema
    class Unrelated:
        def query(self, _):
            return [{'name': 'unrelated'}]
    with pytest.raises(access.DatabaseError) as exc:
        _check_schema(Unrelated())
    assert exc.value.code == 'compat_unsupported'
    with pytest.raises(access.DatabaseError):
        _set_mapping(None, 0x3ffffc00)


def test_other_offsets_and_page_sizes_are_safe(tmp_path):
    path = tmp_path/'unrecognized.db'
    path.write_bytes(b'P'*2048+b'E'*4096)
    with pytest.raises(access.DatabaseError) as exc:
        with access.open_database(path, config={'password':'synthetic'}):
            pass
    assert exc.value.code == 'format'
    with pytest.raises(access.DatabaseError):
        access._config({'password':'synthetic','cipher_page_size':3000})


def _nested_reader(path, config, pipe, gate):
    if not gate.wait(10):
        return
    connection = IsolatedCipherConnection(path, config, lambda: False)
    pipe.send(connection.process.pid)
    pipe.recv()
    connection.close()


def test_analysis_job_cancellation_reaps_nested_reader(encrypted):
    path, cfg, _ = encrypted
    from local_process import WindowsJob
    context = mp.get_context('spawn')
    parent, child = context.Pipe()
    gate = context.Event()
    job = WindowsJob()
    process = context.Process(target=_nested_reader, args=(path, cfg, child, gate))
    try:
        process.start()
        child.close()
        job.assign(process.pid)
        gate.set()
        assert parent.poll(15)
        reader_pid = parent.recv()
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        api.OpenProcess.restype = ctypes.c_void_p
        handle = api.OpenProcess(0x100000 | 0x1000, False, reader_pid)
        assert handle
        job.close()
        process.join(5)
        assert not process.is_alive()
        try:
            api.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            assert api.WaitForSingleObject(handle, 5000) == 0
        finally:
            api.CloseHandle.argtypes = [ctypes.c_void_p]
            api.CloseHandle(handle)
    finally:
        job.close()
        if process.is_alive():
            process.terminate()
        process.join(5)
        process.close()
        parent.close()


def test_pythonw_reader_startup_and_teardown(encrypted):
    path, cfg, _ = encrypted
    # Match the desktop bootstrap: skip the venv launch redirector when changing
    # executable kind from python.exe to pythonw.exe.
    pythonw = Path(getattr(sys, '_base_executable', sys.executable)).with_name('pythonw.exe')
    assert pythonw.is_file()
    mp.set_executable(str(pythonw))
    connection = None
    try:
        connection = IsolatedCipherConnection(path, cfg, lambda: False)
        assert connection.query('SELECT count(*) AS n FROM group_msg_table') == [{'n': '2'}]
    finally:
        if connection is not None:
            connection.close()
        mp.set_executable(sys.executable)


def test_unverified_engine_is_rejected_before_mapping(monkeypatch):
    import isolated_database as isolated
    received = []
    class FakeConnection:
        def __init__(self, *_):
            pass
        def query(self, sql):
            return [{'v':'unverified'}] if sql.startswith('SELECT') else [{'cipher_version':'unverified'}]
        def close(self):
            pass
    class Pipe:
        def send_bytes(self, packet):
            received.append(isolated.json.loads(packet))
        def close(self):
            pass
    monkeypatch.setattr(access, 'CipherConnection', FakeConnection)
    monkeypatch.setattr(isolated, '_silence', lambda: None)
    isolated._worker(Pipe(), SimpleNamespace(wait=lambda _: True), 'synthetic', {})
    assert received == [{'kind':'error','code':'compat_unsupported'}]


def test_spawn_failure_is_safe_and_not_reported_as_corruption(tmp_path, monkeypatch):
    path, cfg = tmp_path/'not-opened.db', {}
    def fail(_):
        raise OSError('synthetic start failure')
    monkeypatch.setattr(mp.context.SpawnProcess, 'start', fail)
    with pytest.raises(access.DatabaseError) as exc:
        IsolatedCipherConnection(path, cfg, lambda: False)
    assert exc.value.code == 'compat_unsupported'


def test_explicit_config_file_is_scoped_per_call(encrypted, tmp_path, monkeypatch):
    import json
    path, cfg, _ = encrypted
    config_path = tmp_path/'explicit.json'
    config_path.write_text(json.dumps(cfg), encoding='utf-8')
    sentinel = str(tmp_path/'unavailable.json')
    monkeypatch.setenv('TEI_CONFIG_FILE', sentinel)
    with access.open_database(path, config_file=config_path) as (connection, _, info):
        assert connection.query('SELECT count(*) AS n FROM group_msg_table') == [{'n':'2'}]
        assert info['readonly']
        assert info['source_hash_status'] == 'not_computed_standard'
        assert info['source_sha256'] is None
    assert os.environ['TEI_CONFIG_FILE'] == sentinel


def test_ambiguous_or_invalid_config_file_is_rejected(tmp_path):
    path, cfg = tmp_path/'opaque.db', {'password':'synthetic'}
    path.write_bytes(b'E'*4096)
    for kwargs in ({'config':cfg,'config_file':tmp_path/'unused.json'}, {'config_file':123}):
        with pytest.raises(access.DatabaseError) as exc:
            with access.open_database(path, **kwargs):
                pass
        assert exc.value.code == 'config_invalid'
