"""Bounded, private, read-only database opening with explicit safe errors."""

import ctypes
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time
from contextlib import contextmanager

SQLITE_MAGIC = b"SQLite format 3\0"
COPY_LIMIT = 4 * 1024**3
COPY_SECONDS = 120
QUERY_SECONDS = 60
CONFIG_PATH = Path(__file__).with_name("config.json")
MESSAGES = {
    "missing": "数据库文件不存在或不可读取。",
    "config_missing": "检测到加密数据库候选，但缺少本地 config.json 或数据库密码。请配置 SQLCipher 后重试。",
    "config_invalid": "本地 SQLCipher 配置格式或参数无效，请检查 config.json。",
    "dll_missing": "SQLCipher DLL 不存在或无法加载（含依赖缺失）；请检查本地 sqlcipher_dll 配置。",
    "dll_invalid": "配置的 DLL 不支持所需的 SQLCipher 接口。",
    "format": "文件不是可识别的 SQLite 或受支持的 NT 前缀数据库。",
    "unlock": "无法按本地密码与 cipher 参数解锁数据库；也可能是文件格式不符或头页损坏。未读取聊天。",
    "corrupt": "数据库页面损坏，未返回部分统计。请使用完整且一致的数据库副本。",
    "schema": "数据库缺少所选格式所需的表或字段，请核对原生 NT / 结构导入选项。",
    "busy": "数据库正在使用或无法建立只读访问，请使用一致的离线副本；无需关闭 QQ。",
    "wal": "检测到非空 WAL / 回滚日志，本次未打开。请使用完整一致的离线副本，不要手动删除日志。",
    "changed": "读取期间原始数据库发生变化，已丢弃结果，请使用一致的离线副本。",
    "copy_limit": "NT 前缀处理超过 4 GiB 副本预算，请提供已去前缀的一致副本。",
    "disk": "临时目录空间不足，无法创建隔离只读副本。",
    "cancelled": "已取消，未返回部分统计。",
    "timeout": "只读准备或查询超过时间预算，请缩小日期范围或使用一致副本。",
    "result_limit": "选区查询结果超过内存预算，请缩小日期范围。",
    "query": "只读查询失败，未返回部分统计。",
}


class DatabaseError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(MESSAGES[code])


def _stamp(path):
    s = path.stat()
    return s.st_size, s.st_mtime_ns


def _config(config):
    if config is None:
        try:
            if not CONFIG_PATH.is_file():
                raise DatabaseError("config_missing")
            if CONFIG_PATH.stat().st_size > 65536:
                raise DatabaseError("config_invalid")
            config = json.loads(CONFIG_PATH.read_text("utf-8-sig"))
        except (OSError, ValueError) as exc:
            if isinstance(exc, DatabaseError):
                raise
            raise DatabaseError("config_invalid") from None
    if not isinstance(config, dict):
        raise DatabaseError("config_invalid")
    cfg = dict(config)
    key = cfg.get("password") or os.environ.get("SQLCIPHER_PASSWORD")
    if not key:
        raise DatabaseError("config_missing")
    if not isinstance(key, str) or "\0" in key or len(key) > 4096:
        raise DatabaseError("config_invalid")
    cfg["password"] = key
    if "sqlcipher_dll" in cfg and (
        not isinstance(cfg["sqlcipher_dll"], str)
        or not cfg["sqlcipher_dll"]
        or "\0" in cfg["sqlcipher_dll"]
    ):
        raise DatabaseError("config_invalid")
    try:
        page = int(cfg.get("cipher_page_size", 4096))
        iterations = int(cfg.get("kdf_iter", 4000))
    except (TypeError, ValueError):
        raise DatabaseError("config_invalid") from None
    if (
        page not in (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
        or not 1 <= iterations <= 10000000
    ):
        raise DatabaseError("config_invalid")
    cfg["cipher_page_size"] = page
    cfg["kdf_iter"] = iterations
    if cfg.get("cipher_hmac_algorithm", "HMAC_SHA1") not in (
        "HMAC_SHA1",
        "HMAC_SHA256",
        "HMAC_SHA512",
    ) or cfg.get("cipher_kdf_algorithm", "PBKDF2_HMAC_SHA512") not in (
        "PBKDF2_HMAC_SHA1",
        "PBKDF2_HMAC_SHA256",
        "PBKDF2_HMAC_SHA512",
    ):
        raise DatabaseError("config_invalid")
    return cfg


def _error(rc, unlock=False):
    code = rc & 255
    if code == 26:
        return DatabaseError("unlock" if unlock else "format")
    if code == 11:
        return DatabaseError("corrupt")
    if code in (5, 6, 8, 14):
        return DatabaseError("busy")
    return DatabaseError("query")


class CipherConnection:
    def __init__(self, path, cfg, cancelled):
        self.db = ctypes.c_void_p()
        self.cancelled = cancelled
        self.deadline = 0
        dll = Path(cfg.get("sqlcipher_dll", "C:/msys64/mingw64/bin/libsqlcipher-0.dll"))
        self.dll_dir = None
        try:
            if not dll.is_file():
                raise OSError()
            if os.name == "nt":
                self.dll_dir = os.add_dll_directory(str(dll.resolve().parent))
            self.lib = ctypes.CDLL(str(dll.resolve()))
        except OSError:
            self.close()
            raise DatabaseError("dll_missing") from None
        try:
            signatures = {
                "sqlite3_open_v2": (
                    [
                        ctypes.c_char_p,
                        ctypes.POINTER(ctypes.c_void_p),
                        ctypes.c_int,
                        ctypes.c_char_p,
                    ],
                    ctypes.c_int,
                ),
                "sqlite3_close": ([ctypes.c_void_p], ctypes.c_int),
                "sqlite3_key": (
                    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int],
                    ctypes.c_int,
                ),
                "sqlite3_prepare_v2": (
                    [
                        ctypes.c_void_p,
                        ctypes.c_char_p,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_void_p),
                        ctypes.c_void_p,
                    ],
                    ctypes.c_int,
                ),
                "sqlite3_step": ([ctypes.c_void_p], ctypes.c_int),
                "sqlite3_finalize": ([ctypes.c_void_p], ctypes.c_int),
                "sqlite3_column_count": ([ctypes.c_void_p], ctypes.c_int),
                "sqlite3_column_name": (
                    [ctypes.c_void_p, ctypes.c_int],
                    ctypes.c_char_p,
                ),
                "sqlite3_column_type": ([ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
                "sqlite3_column_text": (
                    [ctypes.c_void_p, ctypes.c_int],
                    ctypes.c_void_p,
                ),
                "sqlite3_column_bytes": ([ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
                "sqlite3_db_readonly": (
                    [ctypes.c_void_p, ctypes.c_char_p],
                    ctypes.c_int,
                ),
            }
            for name, (args, result) in signatures.items():
                f = getattr(self.lib, name)
                f.argtypes = args
                f.restype = result
            uri = Path(path).resolve().as_uri() + "?mode=ro&immutable=1"
            rc = self.lib.sqlite3_open_v2(
                uri.encode(), ctypes.byref(self.db), 1 | 0x40, None
            )
            if rc:
                raise _error(rc)
            if self.lib.sqlite3_db_readonly(self.db, b"main") != 1:
                raise DatabaseError("busy")
            callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
            self.callback = callback_type(
                lambda _: int(self.cancelled() or time.monotonic() > self.deadline)
            )
            self.lib.sqlite3_progress_handler.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                callback_type,
                ctypes.c_void_p,
            ]
            self.lib.sqlite3_progress_handler(self.db, 1000, self.callback, None)
            key = cfg["password"].encode("utf-8")
            if self.lib.sqlite3_key(self.db, key, len(key)):
                raise DatabaseError("unlock")
            for name, value in [
                ("cipher_page_size", cfg["cipher_page_size"]),
                ("kdf_iter", cfg["kdf_iter"]),
                (
                    "cipher_hmac_algorithm",
                    cfg.get("cipher_hmac_algorithm", "HMAC_SHA1"),
                ),
                (
                    "cipher_kdf_algorithm",
                    cfg.get("cipher_kdf_algorithm", "PBKDF2_HMAC_SHA512"),
                ),
            ]:
                self.query(f"PRAGMA {name} = {value}")
            if not self.query("PRAGMA cipher_version"):
                raise DatabaseError("dll_invalid")
            self.query("PRAGMA query_only=ON")
            self.query("SELECT count(*) AS n FROM sqlite_master", unlock=True)
        except AttributeError:
            self.close()
            raise DatabaseError("dll_invalid") from None
        except BaseException:
            self.close()
            raise

    def query(self, sql, params=(), unlock=False):
        rows = []
        size = 0
        for row in self.iter_query(sql, params, unlock):
            size += sum(len(str(value).encode("utf-8")) for value in row.values())
            if size > 96 * 1024 * 1024:
                raise DatabaseError("result_limit")
            rows.append(row)
        return rows

    def iter_query(self, sql, params=(), unlock=False):
        if params:
            raise DatabaseError("query")
        if self.cancelled():
            raise DatabaseError("cancelled")
        self.deadline = time.monotonic() + QUERY_SECONDS
        statement = ctypes.c_void_p()
        try:
            rc = self.lib.sqlite3_prepare_v2(
                self.db, sql.encode(), -1, ctypes.byref(statement), None
            )
            if rc:
                if self.cancelled():
                    raise DatabaseError("cancelled")
                if time.monotonic() > self.deadline:
                    raise DatabaseError("timeout")
                raise _error(rc, unlock)
            count = self.lib.sqlite3_column_count(statement)
            names = [
                self.lib.sqlite3_column_name(statement, i).decode("utf-8")
                for i in range(count)
            ]
            while True:
                rc = self.lib.sqlite3_step(statement)
                if rc == 101:
                    return
                if rc != 100:
                    if self.cancelled():
                        raise DatabaseError("cancelled")
                    if time.monotonic() > self.deadline:
                        raise DatabaseError("timeout")
                    raise _error(rc, unlock)
                row = {}
                size = 0
                for i, name in enumerate(names):
                    length = self.lib.sqlite3_column_bytes(statement, i)
                    size += length
                    if size > 64 * 1024:
                        raise DatabaseError("result_limit")
                    raw = self.lib.sqlite3_column_text(statement, i)
                    row[name] = (
                        None
                        if self.lib.sqlite3_column_type(statement, i) == 5
                        else (ctypes.string_at(raw, length) if raw else b"").decode(
                            "utf-8", errors="replace"
                        )
                    )
                yield row
        finally:
            if statement:
                self.lib.sqlite3_finalize(statement)

    def close(self):
        if getattr(self, "db", None) and getattr(self, "lib", None):
            self.lib.sqlite3_close(self.db)
            self.db = ctypes.c_void_p()
        if getattr(self, "dll_dir", None):
            self.dll_dir.close()
            self.dll_dir = None


class PlainConnection:
    def __init__(self, path, cancelled):
        self.cancelled = cancelled
        self.db = sqlite3.connect(
            Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True
        )
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA query_only=ON")
        self.deadline = 0
        self.db.set_progress_handler(
            lambda: int(cancelled() or time.monotonic() > self.deadline), 1000
        )

    def query(self, sql, params=()):
        return list(self.iter_query(sql, params))

    def iter_query(self, sql, params=()):
        if self.cancelled():
            raise DatabaseError("cancelled")
        self.deadline = time.monotonic() + QUERY_SECONDS
        try:
            for row in self.db.execute(sql, params):
                yield dict(row)
        except sqlite3.Error as exc:
            if self.cancelled():
                raise DatabaseError("cancelled") from None
            if time.monotonic() > self.deadline:
                raise DatabaseError("timeout") from None
            raise _error(getattr(exc, "sqlite_errorcode", 1)) from None

    def close(self):
        self.db.close()


def _copy_payload(source, target, offset, cancelled, progress):
    size = source.stat().st_size - offset
    if size > COPY_LIMIT:
        raise DatabaseError("copy_limit")
    if shutil.disk_usage(target.parent).free < size + 64 * 1024 * 1024:
        raise DatabaseError("disk")
    started = time.monotonic()
    copied = 0
    last_progress = -1
    try:
        with source.open("rb") as reader, target.open("xb") as writer:
            reader.seek(offset)
            while chunk := reader.read(1024 * 1024):
                if cancelled():
                    raise DatabaseError("cancelled")
                if time.monotonic() - started > COPY_SECONDS:
                    raise DatabaseError("timeout")
                if copied + len(chunk) > size or copied + len(chunk) > COPY_LIMIT:
                    raise DatabaseError("changed")
                writer.write(chunk)
                copied += len(chunk)
                current = 10 + int(15 * copied / max(size, 1))
                if current != last_progress:
                    last_progress = current
                    progress(current, "正在准备隔离的 NT 只读副本…")
    except OSError:
        raise DatabaseError("disk") from None


@contextmanager
def open_database(
    path,
    force_encrypted=False,
    config=None,
    scratch_dir=None,
    cancelled=lambda: False,
    progress=lambda n, t: None,
):
    source = Path(path)
    conn = None
    temporary = None
    try:
        try:
            stamp = _stamp(source)
            with source.open("rb") as stream:
                head = stream.read(1040)
        except OSError:
            raise DatabaseError("missing") from None
        if cancelled():
            raise DatabaseError("cancelled")
        if len(head) < 16:
            raise DatabaseError("format")
        sidecars = [Path(str(source) + suffix) for suffix in ("-wal", "-journal")]
        if any(p.exists() and p.stat().st_size for p in sidecars):
            raise DatabaseError("wal")
        plain = head[:16] == SQLITE_MAGIC
        offset = 0
        if not plain and head[1024:1040] == SQLITE_MAGIC:
            plain = True
            offset = 1024
        cfg = None
        if not plain:
            cfg = _config(config)
            page = cfg["cipher_page_size"]
            if stamp[0] % page == 1024 and page > 1024:
                offset = 1024
            elif stamp[0] % page:
                raise DatabaseError("format")
        elif force_encrypted:
            # Manual SQLCipher is an override for opaque files, not a reason to break plaintext.
            progress(10, "检测到普通 SQLite，按只读明文打开。")
        selected = source
        if offset:
            temporary = tempfile.TemporaryDirectory(
                prefix="tei-payload-", dir=scratch_dir
            )
            selected = Path(temporary.name) / "payload.db"
            _copy_payload(source, selected, offset, cancelled, progress)
            if _stamp(source) != stamp:
                raise DatabaseError("changed")
        progress(27, "正在核验只读数据库格式与本地解密配置…")
        conn = (
            PlainConnection(selected, cancelled)
            if plain
            else CipherConnection(selected, cfg, cancelled)
        )
        conn.query("SELECT count(*) AS n FROM sqlite_master")
        yield (
            conn,
            not plain,
            {
                "database_format": ("nt_prefix_" if offset else "")
                + ("sqlite" if plain else "sqlcipher"),
                "readonly": True,
            },
        )
        if _stamp(source) != stamp or any(
            p.exists() and p.stat().st_size for p in sidecars
        ):
            raise DatabaseError("changed")
    except OSError:
        raise DatabaseError("missing") from None
    finally:
        if conn is not None:
            conn.close()
        if temporary is not None:
            temporary.cleanup()
