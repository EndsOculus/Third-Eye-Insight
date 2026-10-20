"""Version-gated NT lock-page compatibility, confined to a disposable worker.

No DLL patching. SQLITE_TESTCTRL_PENDING_BYTE is a test interface, not a stable
per-connection API: it is used only in this Windows worker on verified versions,
after NT schema/layout checks and with every database handle closed. Never call
it in the desktop/analysis process or share this worker with other databases.
"""
import ctypes
import json
import multiprocessing as mp
import os
import threading
import time

MAX_PACKET = 256 * 1024
DEFAULT_PENDING = 0x40000000
NT_PENDING = DEFAULT_PENDING - 1024
SUPPORTED = ("3.46.1", "4.6.1 community")


def _send(pipe, value):
    packet = json.dumps(value, ensure_ascii=True).encode("ascii")
    if len(packet) > MAX_PACKET:
        raise ValueError("packet_limit")
    pipe.send_bytes(packet)


def _receive(pipe):
    return json.loads(pipe.recv_bytes(MAX_PACKET))


def _silence():
    import logging
    logging.disable(logging.CRITICAL)
    sink = os.open(os.devnull, os.O_WRONLY)
    for fd in (1, 2):
        os.dup2(sink, fd)
    if os.name == "nt":
        import msvcrt
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.SetStdHandle.argtypes = [ctypes.c_ulong, ctypes.c_void_p]
        for number in (-11, -12):
            api.SetStdHandle(number & 0xffffffff, msvcrt.get_osfhandle(1 if number == -11 else 2))
    if sink not in (1, 2):
        os.close(sink)


def _check_schema(connection):
    from database_access import DatabaseError
    # Opening is mode-independent. Require at least one usable NT table; the
    # selected-mode adapter validates its own table again before reading rows.
    # In particular, a reduced/unrelated c2c table must not prevent group reads,
    # and 40001 is an optional message key in interaction_data.
    supported = False
    for table, peer in (("group_msg_table", "40027"), ("c2c_msg_table", "40030")):
        columns = {r["name"] for r in connection.query(f"PRAGMA table_info({table})")}
        supported |= {peer, "40033", "40050", "40800"} <= columns
    if not supported:
        raise DatabaseError("compat_unsupported")
    page = next(iter(connection.query("PRAGMA page_size")[0].values()))
    if int(page) != 4096:
        raise DatabaseError("compat_unsupported")


def _set_mapping(lib, value):
    from database_access import DatabaseError
    if mp.current_process().name != "tei-nt-readonly" or value != NT_PENDING:
        raise DatabaseError("compat_unsupported")
    try:
        control = lib.sqlite3_test_control
        control.restype = ctypes.c_int
        if control(ctypes.c_int(11), ctypes.c_uint(0)) != DEFAULT_PENDING:
            raise DatabaseError("compat_unsupported")
        previous = control(ctypes.c_int(11), ctypes.c_uint(value))
        if previous != DEFAULT_PENDING or control(ctypes.c_int(11), ctypes.c_uint(0)) != value:
            raise DatabaseError("compat_unsupported")
        return previous
    except AttributeError:
        raise DatabaseError("compat_unsupported") from None


def _worker(pipe, gate, path, config):
    # The owner assigns the Windows job before this gate opens. No DB/DLL work
    # can occur before containment, including when the desktop is cancelled.
    if not gate.wait(15):
        return
    _silence()
    from database_access import CipherConnection, DatabaseError
    connection = None
    lib = None
    previous = None
    try:
        connection = CipherConnection(path, config, lambda: False)
        versions = (connection.query("SELECT sqlite_version() AS v")[0]["v"],
                    connection.query("PRAGMA cipher_version")[0]["cipher_version"])
        if versions != SUPPORTED:
            raise DatabaseError("compat_unsupported")
        _check_schema(connection)
        lib = connection.lib
        connection.close()
        connection = None
        previous = _set_mapping(lib, NT_PENDING)
        connection = CipherConnection(path, config, lambda: False)
        connection.query("PRAGMA temp_store=MEMORY")
        _send(pipe, {"kind": "ready", "versions": versions})
        while True:
            command = _receive(pipe)
            if command == {"kind": "close"}:
                return
            if (not isinstance(command, dict) or command.get("kind") != "query"
                    or not isinstance(command.get("sql"), str) or len(command["sql"]) > 32768):
                raise DatabaseError("query")
            sql = command["sql"]
            # Internal adapter protocol, not a general SQL server. query_only
            # also blocks mutating WITH statements; ATTACH/extension operations
            # are never accepted. PRAGMAs are restricted to metadata reads.
            first = sql.lstrip().split(None, 1)[0].upper()
            pragma = sql.strip().lower()
            if first not in ("SELECT", "WITH") and not (
                pragma.startswith("pragma table_info(") and pragma.endswith(")")
                and ";" not in pragma
            ):
                raise DatabaseError("query")
            rows = []
            size = 0
            for row in connection.iter_query(sql):
                row_size = len(json.dumps(row, ensure_ascii=True))
                if rows and (len(rows) >= 32 or size + row_size > 96 * 1024):
                    _send(pipe, {"kind": "rows", "rows": rows})
                    if _receive(pipe) != {"kind": "next"}:
                        return
                    rows, size = [], 0
                rows.append(row)
                size += row_size
            if rows:
                _send(pipe, {"kind": "rows", "rows": rows})
                if _receive(pipe) != {"kind": "next"}:
                    return
            _send(pipe, {"kind": "done"})
    except DatabaseError as exc:
        try:
            _send(pipe, {"kind": "error", "code": exc.code})
        except (OSError, ValueError):
            pass
    except BaseException:
        try:
            _send(pipe, {"kind": "error", "code": "query"})
        except BaseException:
            pass
    finally:
        if connection is not None:
            connection.close()
        if previous is not None:
            lib.sqlite3_test_control(ctypes.c_int(11), ctypes.c_uint(previous))
        pipe.close()


class IsolatedCipherConnection:
    """Synchronous bounded iterator; each instance owns exactly one child job."""
    def __init__(self, path, config, cancelled):
        from database_access import DatabaseError
        if os.name != "nt":
            raise DatabaseError("compat_unsupported")
        from local_process import WindowsJob
        self.cancelled = cancelled
        self.closed = False
        self.busy = threading.Lock()
        self.process = None
        self.pipe = None
        self.job = None
        child = None
        try:
            self.job = WindowsJob()
            context = mp.get_context("spawn")
            self.pipe, child = context.Pipe()
            gate = context.Event()
            self.process = context.Process(target=_worker, args=(child, gate, str(path), config), name="tei-nt-readonly")
            self.process.start()
            child.close()
            self.job.assign(self.process.pid)
            gate.set()
            ready = self._wait(time.monotonic() + 20)
            if ready.get("kind") != "ready":
                raise DatabaseError("query")
            self.versions = tuple(ready["versions"])
        except BaseException as exc:
            if child is not None:
                child.close()
            self.close()
            if isinstance(exc, Exception) and not isinstance(exc, DatabaseError):
                raise DatabaseError("compat_unsupported") from None
            raise

    def _wait(self, deadline):
        from database_access import DatabaseError, MESSAGES
        while True:
            if self.cancelled():
                raise DatabaseError("cancelled")
            if time.monotonic() > deadline:
                raise DatabaseError("timeout")
            if self.pipe.poll(.03):
                try:
                    result = _receive(self.pipe)
                except (OSError, ValueError, EOFError):
                    raise DatabaseError("query") from None
                if not isinstance(result, dict):
                    raise DatabaseError("query")
                if result.get("kind") == "error":
                    raise DatabaseError(result.get("code") if result.get("code") in MESSAGES else "query")
                return result
            if not self.process.is_alive():
                raise DatabaseError("query")

    def query(self, sql, params=()):
        from database_access import DatabaseError
        rows, size = [], 0
        for row in self.iter_query(sql, params):
            size += len(json.dumps(row, ensure_ascii=True))
            if size > 96 * 1024 * 1024:
                raise DatabaseError("result_limit")
            rows.append(row)
        return rows

    def iter_query(self, sql, params=()):
        from database_access import DatabaseError, QUERY_SECONDS
        if self.closed or params:
            raise DatabaseError("query")
        if not self.busy.acquire(blocking=False):
            raise DatabaseError("busy")
        done = False
        try:
            _send(self.pipe, {"kind": "query", "sql": sql})
            deadline = time.monotonic() + QUERY_SECONDS + 2
            while True:
                result = self._wait(deadline)
                if result.get("kind") == "done":
                    done = True
                    return
                if result.get("kind") != "rows" or not isinstance(result.get("rows"), list):
                    raise DatabaseError("query")
                yield from result["rows"]
                _send(self.pipe, {"kind": "next"})
        except (OSError, ValueError, EOFError) as exc:
            if isinstance(exc, DatabaseError):
                raise
            raise DatabaseError("query") from None
        finally:
            if not done:
                self.close()
            self.busy.release()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.pipe is not None:
            try:
                _send(self.pipe, {"kind": "close"})
            except (OSError, ValueError):
                pass
            self.pipe.close()
        if self.job is not None:
            self.job.close()
        if self.process is not None:
            if self.process.pid is None:
                self.process.close()
                return
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=5)
            if not self.process.is_alive():
                self.process.close()
