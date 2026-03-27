"""
extract_chat_data.py
--------------------
从 SQLite（明文或 SQLCipher 加密）或远程 PostgreSQL 数据库中提取聊天数据。
加密数据库配置通过 config.json 指定。
"""

import sqlite3
import json
import os
import ctypes
import pandas as pd


def _load_config() -> dict:
    if os.path.exists('config.json'):
        with open('config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def clean_message(text: str) -> str:
    try:
        return text.encode('gbk', errors='replace').decode('gbk')
    except Exception:
        return text


def _sqlcipher_query(db_path: str, query: str, config: dict) -> pd.DataFrame:
    """使用 ctypes 调用 SQLCipher DLL 查询加密数据库，返回 DataFrame。"""
    dll_path = config.get('sqlcipher_dll', 'C:/msys64/mingw64/bin/libsqlcipher-0.dll')
    try:
        lib = ctypes.CDLL(dll_path)
    except OSError as e:
        print(f"[ERROR] 无法加载 SQLCipher DLL ({dll_path}): {e}")
        return pd.DataFrame()

    lib.sqlite3_open.restype = ctypes.c_int
    lib.sqlite3_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.sqlite3_exec.restype = ctypes.c_int
    lib.sqlite3_exec.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                  ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.POINTER(ctypes.c_char_p)]
    lib.sqlite3_close.restype = ctypes.c_int

    db = ctypes.c_void_p()
    if lib.sqlite3_open(db_path.encode(), ctypes.byref(db)) != 0:
        print("[ERROR] 无法打开加密数据库文件。")
        return pd.DataFrame()

    pragma_sql = "; ".join([
        f"PRAGMA key = '{config.get('password', '')}'",
        f"PRAGMA cipher_page_size = {config.get('cipher_page_size', 4096)}",
        f"PRAGMA kdf_iter = {config.get('kdf_iter', 4000)}",
        f"PRAGMA cipher_hmac_algorithm = {config.get('cipher_hmac_algorithm', 'HMAC_SHA1')}",
        f"PRAGMA cipher_kdf_algorithm = {config.get('cipher_kdf_algorithm', 'PBKDF2_HMAC_SHA512')}",
    ]) + ";"

    errmsg = ctypes.c_char_p()
    if lib.sqlite3_exec(db, pragma_sql.encode(), None, None, ctypes.byref(errmsg)) != 0:
        print(f"[ERROR] 加密参数设置失败: {errmsg.value}")
        lib.sqlite3_close(db)
        return pd.DataFrame()

    rows = []
    col_names = []

    CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_char_p),
                                  ctypes.POINTER(ctypes.c_char_p))

    def _cb(_, ncols, values, cols):
        if not col_names:
            for i in range(ncols):
                col_names.append(cols[i].decode('utf-8') if cols[i] else f'col{i}')
        rows.append([
            values[i].decode('utf-8', errors='replace') if values[i] else None
            for i in range(ncols)
        ])
        return 0

    cb = CALLBACK(_cb)
    rc = lib.sqlite3_exec(db, query.encode(), cb, None, ctypes.byref(errmsg))
    lib.sqlite3_close(db)

    if rc != 0:
        print(f"[ERROR] SQLCipher 查询失败 (rc={rc}): {errmsg.value}")
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=col_names) if rows else pd.DataFrame(columns=col_names or [])


def _build_local_query(mode: str, identifier: int, encrypted: bool) -> str:
    """构建本地 SQLite 查询语句。加密模式下直接嵌入参数（整数安全）。"""
    table = "group_msg_table" if mode == "group" else "c2c_msg_table"
    id_col = '"40027"' if mode == "group" else '"40033"'
    param = str(identifier) if encrypted else "?"
    return f"""
        SELECT
            "40033" AS sender_id,
            "40090" AS group_nickname,
            "40093" AS qq_name,
            "40080" AS content,
            "40050" AS timestamp
        FROM {table}
        WHERE {id_col} = {param}
          AND "40011" = 2
          AND "40012" = 1
          AND content IS NOT NULL
          AND TRIM(content) <> ''
          AND "40033" NOT IN (2854196310, 10000)
        ORDER BY "40050"
    """


def _postprocess(df: pd.DataFrame, mode: str, identifier: int) -> pd.DataFrame:
    """时间戳转换、昵称合并、内容清洗。"""
    if df.empty:
        print(f"[INFO] {mode} 模式下，标识符 {identifier} 未提取到有效数据。")
        return df

    print("原始时间戳数据：", df['timestamp'].head())
    df['sender_id'] = df['sender_id'].astype(str)
    try:
        df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp'], errors='coerce'), unit='s', errors='coerce', utc=True)
        converted = df['timestamp'].dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
        if not converted.isna().all():
            df['timestamp'] = converted
        else:
            print("[WARN] 时间转换失败，保留 UTC 时间")
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    except Exception as e:
        print(f"[WARN] 时间戳转换失败: {e}")
    df['content'] = df['content'].apply(clean_message)
    df['sender_nickname'] = df['group_nickname'].fillna('').str.strip()
    mask = df['sender_nickname'] == ''
    df.loc[mask, 'sender_nickname'] = df.loc[mask, 'qq_name']
    df.drop(columns=['group_nickname', 'qq_name'], inplace=True)
    return df


def extract_chat_data(db_path: str, identifier: int, mode: str = "group",
                      remote: bool = False, cipher_config: dict = None) -> pd.DataFrame:
    if mode not in ("group", "c2c"):
        print(f"[ERROR] 未知的 mode: {mode}")
        return pd.DataFrame()

    # 远程 PostgreSQL
    if remote:
        from sqlalchemy import create_engine
        try:
            engine = create_engine(db_path)
        except Exception as e:
            print(f"[ERROR] 无法连接远程数据库: {e}")
            return pd.DataFrame()
        if mode == "group":
            query = """
                SELECT (message->>'sender_id') AS sender_id,
                       (message->>'sender_nickname') AS sender_nickname,
                       plain_text AS content, time AS timestamp
                FROM public.nonebot_plugin_chatrecorder_messagerecord
                WHERE type = 'message'
                  AND plain_text IS NOT NULL AND TRIM(plain_text) <> ''
                  AND (message->>'group_id')::bigint = %s
                  AND (message->>'sender_id')::bigint NOT IN (2854196310, 10000)
                ORDER BY time
            """
        else:
            query = """
                SELECT (message->>'sender_id') AS sender_id,
                       (message->>'sender_nickname') AS sender_nickname,
                       plain_text AS content, time AS timestamp
                FROM public.nonebot_plugin_chatrecorder_messagerecord
                WHERE type = 'message'
                  AND plain_text IS NOT NULL AND TRIM(plain_text) <> ''
                  AND (message->>'sender_id')::bigint = %s
                  AND (message->>'sender_id')::bigint NOT IN (2854196310, 10000)
                ORDER BY time
            """
        try:
            df = pd.read_sql_query(query, engine, params=(identifier,))
        except Exception as e:
            print(f"[ERROR] 执行 SQL 查询失败: {e}")
            return pd.DataFrame()
        return df

    # 加密本地 SQLite
    config = cipher_config if cipher_config is not None else _load_config()
    if config.get('encrypted', False):
        query = _build_local_query(mode, identifier, encrypted=True)
        df = _sqlcipher_query(db_path, query, config)
        return _postprocess(df, mode, identifier)

    # 明文本地 SQLite
    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        print(f"[ERROR] 无法连接数据库: {e}")
        return pd.DataFrame()
    query = _build_local_query(mode, identifier, encrypted=False)
    try:
        df = pd.read_sql_query(query, conn, params=(identifier,))
    except Exception as e:
        print(f"[ERROR] 执行 SQL 查询失败: {e}")
        return pd.DataFrame()
    finally:
        conn.close()
    return _postprocess(df, mode, identifier)


if __name__ == "__main__":
    data = extract_chat_data("nt_msg.clean.db", 98765432, mode="group")
    print(f"群聊模式下提取到 {len(data)} 条消息记录")
