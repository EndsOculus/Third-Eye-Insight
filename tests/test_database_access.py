import hashlib
from pathlib import Path
import sqlite3
import pytest
import database_access as access
from desktop_core import Request, analyze, InputError
from tests.interaction_fixtures import make_db, msg


def fixture_db(tmp_path, prefix=False):
    path = tmp_path / '\u4e2d\u6587 space.sqlite'
    make_db(path, [msg(1, 0, 'a'), msg(2, 10, 'b')], source_format='nt')
    if prefix:
        path.write_bytes(b'N' * 1024 + path.read_bytes())
    return path


@pytest.mark.parametrize('prefix', [False, True])
@pytest.mark.parametrize('manual', [False, True])
def test_plain_auto_and_manual_readonly(tmp_path, monkeypatch, prefix, manual):
    path = fixture_db(tmp_path, prefix)
    before = (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).digest())
    monkeypatch.setattr(access, 'CONFIG_PATH', tmp_path / 'absent.json')
    scratch = tmp_path / 'scratch'; scratch.mkdir()
    result = analyze(Request(db=str(path), identifier='99', encrypted=manual), scratch_dir=scratch)
    assert result['rows'] == 2
    assert result['source']['database_format'] == ('nt_prefix_' if prefix else '') + 'sqlite'
    assert not list(scratch.iterdir())
    assert before == (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).digest())
    assert not Path(str(path)+'-wal').exists()
    assert not Path(str(path)+'-shm').exists()


@pytest.mark.parametrize('prefix', [False, True])
@pytest.mark.parametrize('manual', [False, True])
def test_encrypted_routing_config_and_cleanup(tmp_path, monkeypatch, prefix, manual):
    path = tmp_path / 'opaque.any'; payload=b'E'*4096
    path.write_bytes((b'N'*1024 if prefix else b'') + payload)
    cfg={'password':'synthetic-only', 'cipher_page_size':4096, 'kdf_iter':4000}
    seen=[]
    class Cipher:
        def __init__(self, selected, config, cancelled):
            assert Path(selected).read_bytes()==payload
            assert config['password']=='synthetic-only'
            seen.append(Path(selected))
        def query(self, sql): return [{'n':'1'}]
        def close(self): seen.append('closed')
    monkeypatch.setattr(access,'CipherConnection',Cipher)
    with access.open_database(path, force_encrypted=manual, config=cfg, scratch_dir=tmp_path) as (_, encrypted, info):
        assert encrypted and info['readonly']
        assert info['database_format']==('nt_prefix_' if prefix else '')+'sqlcipher'
    assert seen[-1]=='closed'
    assert not prefix or not seen[0].exists()


@pytest.mark.parametrize('kind,code', [('absent','config_missing'),('invalid','config_invalid'),('wrongdll','dll_missing')])
def test_config_errors_are_safe(tmp_path, monkeypatch, kind, code):
    path=tmp_path/'opaque.db';path.write_bytes(b'E'*4096)
    config=tmp_path/'config.json'
    monkeypatch.setattr(access,'CONFIG_PATH',config)
    if kind=='invalid':config.write_text('{invalid',encoding='utf-8')
    if kind=='wrongdll':config.write_text('{"password":"synthetic-secret","sqlcipher_dll":"missing-private-path"}',encoding='utf-8')
    with pytest.raises(access.DatabaseError) as exc:
        with access.open_database(path):pass
    assert exc.value.code==code
    assert 'synthetic-secret' not in str(exc.value) and 'missing-private-path' not in str(exc.value)


def test_format_corruption_and_schema(tmp_path):
    path=tmp_path/'bad.db';path.write_bytes(b'short')
    with pytest.raises(access.DatabaseError) as exc:
        with access.open_database(path):pass
    assert exc.value.code=='format'
    path.write_bytes(access.SQLITE_MAGIC+b'X'*4080)
    with pytest.raises(access.DatabaseError):
        with access.open_database(path):pass
    path.unlink()
    with sqlite3.connect(path) as db:db.execute('create table unrelated(x)')
    with pytest.raises(InputError):analyze(Request(db=str(path),identifier='99'))


@pytest.mark.parametrize('failure', ['cancel','space','size','timeout','query'])
def test_prefix_failure_cleanup(tmp_path, monkeypatch, failure):
    path=fixture_db(tmp_path,True);scratch=tmp_path/'scratch';scratch.mkdir()
    before=path.read_bytes();calls=[]
    def cancelled():
        calls.append(1)
        return failure=='cancel' and len(calls)>1
    if failure=='space':monkeypatch.setattr(access.shutil,'disk_usage',lambda _:type('Usage',(),{'free':0})())
    if failure=='size':monkeypatch.setattr(access,'COPY_LIMIT',1)
    if failure=='timeout':monkeypatch.setattr(access,'COPY_SECONDS',-1)
    with pytest.raises(access.DatabaseError) as exc:
        with access.open_database(path,scratch_dir=scratch,cancelled=cancelled) as (db,_,__):
            db.query('SELECT * FROM missing_table')
    assert exc.value.code=={'cancel':'cancelled','space':'disk','size':'copy_limit','timeout':'timeout','query':'query'}[failure]
    assert not list(scratch.iterdir()) and path.read_bytes()==before


def test_wal_refusal_and_source_change(tmp_path):
    path=fixture_db(tmp_path)
    wal=Path(str(path)+'-wal');wal.write_bytes(b'synthetic')
    with pytest.raises(access.DatabaseError) as exc:
        with access.open_database(path):pass
    assert exc.value.code=='wal' and wal.read_bytes()==b'synthetic'
    wal.unlink()
    with pytest.raises(access.DatabaseError) as exc:
        with access.open_database(path):
            import os
            stamp=path.stat()
            os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns+1000000000))
    assert exc.value.code=='changed'


def test_query_timeout_and_progress_bound(tmp_path, monkeypatch):
    path=fixture_db(tmp_path,True);events=[]
    with access.open_database(path,progress=lambda n,t:events.append(n)) as (db,_,__):
        monkeypatch.setattr(access,'QUERY_SECONDS',-1)
        with pytest.raises(access.DatabaseError) as exc:
            db.query('WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<100000) SELECT sum(x) FROM n')
        assert exc.value.code=='timeout'
    assert len(events)<=17

@pytest.mark.parametrize('suffix', ['-journal','-wal'])
def test_nonempty_sidecars_at_open_and_during_read(tmp_path,suffix):
    path=fixture_db(tmp_path)
    sidecar=Path(str(path)+suffix)
    # Conservatively reject even a non-hot nonempty sidecar; no recovery writes.
    sidecar.write_bytes(b'synthetic rollback or WAL bytes')
    before=(path.read_bytes(),sidecar.read_bytes())
    with pytest.raises(access.DatabaseError) as exc:
        with access.open_database(path):pass
    assert exc.value.code=='wal'
    assert before==(path.read_bytes(),sidecar.read_bytes())
    sidecar.unlink()
    with pytest.raises(access.DatabaseError) as exc:
        with access.open_database(path):sidecar.write_bytes(b'new journal')
    assert exc.value.code=='changed'


def test_cipher_embedded_nul_is_not_truncated():
    import ctypes,time
    content=b'x\0a';buffer=ctypes.create_string_buffer(content)
    class Lib:
        steps=iter([100,101])
        def sqlite3_prepare_v2(self,*args):return 0
        def sqlite3_column_count(self,*args):return 1
        def sqlite3_column_name(self,*args):return b'message_id'
        def sqlite3_step(self,*args):return next(self.steps)
        def sqlite3_column_bytes(self,*args):return len(content)
        def sqlite3_column_text(self,*args):return ctypes.addressof(buffer)
        def sqlite3_column_type(self,*args):return 3
    connection=access.CipherConnection.__new__(access.CipherConnection)
    connection.lib=Lib();connection.db=None;connection.cancelled=lambda:False
    assert connection.query('synthetic')==[{'message_id':'x\0a'}]
