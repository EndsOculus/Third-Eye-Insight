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
import re
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    if os.path.exists('config.json'):
        with open('config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def clean_message(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = ''.join(ch for ch in text if ch >= ' ' or ch in '\n\t')
    matches = re.findall(r'[\u4e00-\u9fffA-Za-z0-9][\u4e00-\u9fffA-Za-z0-9\s.,!?;:，。！？；：“”‘’（）()《》、…+\-_/]*', text)
    if matches:
        text = max(matches, key=len).strip()
    try:
        return text.encode('gbk', errors='replace').decode('gbk')
    except Exception:
        return text


def _resolve_local_c2c_peer_ids_sqlcipher(db_path: str, identifier: int, config: dict):
    query = f"""
        SELECT DISTINCT "40030" AS peer_id
        FROM c2c_msg_table
        WHERE "40033" = {identifier}
          AND "40030" IS NOT NULL
    """
    df = _sqlcipher_query(db_path, query, config)
    return [str(v) for v in df["peer_id"].dropna().tolist()] if not df.empty else []


def _resolve_local_c2c_peer_ids_sqlite(conn, identifier: int):
    query = """
        SELECT DISTINCT "40030" AS peer_id
        FROM c2c_msg_table
        WHERE "40033" = ?
          AND "40030" IS NOT NULL
    """
    df = pd.read_sql_query(query, conn, params=(identifier,))
    return [str(v) for v in df["peer_id"].dropna().tolist()] if not df.empty else []


def _rank_name_candidates(rows):
    candidates = []
    for raw_name, cnt in rows:
        name = "" if raw_name is None else str(raw_name).strip()
        if not name or name.lower() == "nan" or name.isdigit():
            continue
        candidates.append((name, int(cnt)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[1], len(item[0])), reverse=True)
    return candidates[0][0]


def _format_unix_timestamp_local(ts_value) -> str:
    try:
        ts_numeric = pd.to_numeric(ts_value, errors='coerce')
        if pd.isna(ts_numeric):
            return "未知时间"
        dt = pd.to_datetime(ts_numeric, unit='s', utc=True).tz_convert('Asia/Shanghai').tz_localize(None)
        return dt.strftime("%Y/%m/%d %H:%M:%S")
    except Exception:
        return "未知时间"


def build_display_name_map(db_path: str, sender_ids, remote: bool = False, cipher_config: dict = None) -> dict:
    sender_ids = [str(s) for s in sender_ids]
    if remote or not sender_ids:
        return {}
    config = cipher_config if cipher_config is not None else _load_config()
    result = {}
    if config.get('encrypted', False):
        id_filter = ", ".join(sender_ids)
        query = f"""
            SELECT matched_id, name, cnt
            FROM (
                SELECT "40033" AS matched_id,
                       COALESCE(NULLIF(TRIM("40093"), ''), NULL) AS name,
                       COUNT(*) AS cnt
                FROM c2c_msg_table
                WHERE "40033" IN ({id_filter})
                GROUP BY "40033", name
            )
            ORDER BY matched_id, cnt DESC
        """
        df = _sqlcipher_query(db_path, query, config, verbose=False)
    else:
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            placeholders = ", ".join(["?"] * len(sender_ids))
            query = f"""
                SELECT matched_id, name, cnt
                FROM (
                    SELECT "40033" AS matched_id,
                           COALESCE(NULLIF(TRIM("40093"), ''), NULL) AS name,
                           COUNT(*) AS cnt
                    FROM c2c_msg_table
                    WHERE "40033" IN ({placeholders})
                    GROUP BY "40033", name
                )
                ORDER BY matched_id, cnt DESC
            """
            df = pd.read_sql_query(query, conn, params=tuple(sender_ids))
        except Exception:
            return {}
        finally:
            if conn is not None:
                conn.close()
    if df.empty:
        return {}
    for matched_id, group in df.groupby('matched_id'):
        best = _rank_name_candidates(group[['name', 'cnt']].itertuples(index=False, name=None))
        if best:
            result[str(matched_id)] = best
    return result


def _sqlcipher_query_status(db_path: str, query: str, config: dict, verbose: bool = True):
    """使用 ctypes 调用 SQLCipher DLL 查询加密数据库，返回 (DataFrame, rc, errmsg)。"""
    dll_path = config.get('sqlcipher_dll', 'C:/msys64/mingw64/bin/libsqlcipher-0.dll')
    try:
        lib = ctypes.CDLL(dll_path)
    except OSError as e:
        if verbose:
            logger.error("无法加载 SQLCipher DLL (%s): %s", dll_path, e)
        return pd.DataFrame(), -1, str(e)

    lib.sqlite3_open.restype = ctypes.c_int
    lib.sqlite3_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.sqlite3_exec.restype = ctypes.c_int
    lib.sqlite3_exec.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                  ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.POINTER(ctypes.c_char_p)]
    lib.sqlite3_close.restype = ctypes.c_int

    db = ctypes.c_void_p()
    if lib.sqlite3_open(db_path.encode(), ctypes.byref(db)) != 0:
        if verbose:
            logger.error("无法打开加密数据库文件。")
        return pd.DataFrame(), -1, "无法打开加密数据库文件"

    env_password = os.environ.get('SQLCIPHER_PASSWORD')
    cfg_password = config.get('password', '')
    if cfg_password:
        password = cfg_password
        if env_password and env_password != cfg_password and verbose:
            logger.warning("检测到 SQLCIPHER_PASSWORD 与 config.json 不一致，已优先使用 config.json 中的密码")
    else:
        password = env_password or ''
    cfg_page_size = config.get('cipher_page_size', 4096)
    cfg_kdf_iter = config.get('kdf_iter', 4000)
    cfg_hmac = config.get('cipher_hmac_algorithm', 'HMAC_SHA1')
    cfg_kdf_algo = config.get('cipher_kdf_algorithm', 'PBKDF2_HMAC_SHA512')

    pragma_profiles = [
        [
            f"PRAGMA key = '{password}'",
            f"PRAGMA cipher_page_size = {cfg_page_size}",
            f"PRAGMA kdf_iter = {cfg_kdf_iter}",
            f"PRAGMA cipher_hmac_algorithm = {cfg_hmac}",
            f"PRAGMA cipher_kdf_algorithm = {cfg_kdf_algo}",
        ],
        [
            "PRAGMA cipher_compatibility = 4",
            f"PRAGMA key = '{password}'",
        ],
        [
            "PRAGMA cipher_compatibility = 3",
            f"PRAGMA key = '{password}'",
        ],
        [
            "PRAGMA cipher_compatibility = 2",
            f"PRAGMA key = '{password}'",
        ],
        [
            "PRAGMA cipher_compatibility = 1",
            f"PRAGMA key = '{password}'",
        ],
    ]

    # 去重，避免配置与兼容档位重复
    unique_profiles = []
    seen = set()
    for profile in pragma_profiles:
        key = tuple(profile)
        if key in seen:
            continue
        seen.add(key)
        unique_profiles.append(profile)

    errmsg = ctypes.c_char_p()
    last_errmsg = ""
    profile_ok = False
    for idx, profile in enumerate(unique_profiles, start=1):
        pragma_sql = "; ".join(profile) + ";"
        if lib.sqlite3_exec(db, pragma_sql.encode(), None, None, ctypes.byref(errmsg)) != 0:
            last_errmsg = errmsg.value.decode('utf-8', errors='replace') if errmsg.value else ""
            continue
        probe_sql = "SELECT count(*) FROM sqlite_master;"
        probe_rc = lib.sqlite3_exec(db, probe_sql.encode(), None, None, ctypes.byref(errmsg))
        if probe_rc == 0:
            profile_ok = True
            if verbose and idx > 1:
                logger.warning("SQLCipher 使用兼容参数档位 #%d 解密成功", idx)
            break
        last_errmsg = errmsg.value.decode('utf-8', errors='replace') if errmsg.value else ""

    if not profile_ok:
        if verbose:
            logger.error("加密参数设置失败: %s", last_errmsg)
        lib.sqlite3_close(db)
        return pd.DataFrame(), -1, last_errmsg

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
        if verbose:
            logger.error("SQLCipher 查询失败 (rc=%d): %s", rc, errmsg.value)
        return pd.DataFrame(), rc, errmsg.value.decode('utf-8', errors='replace') if errmsg.value else ""

    return pd.DataFrame(rows, columns=col_names) if rows else pd.DataFrame(columns=col_names or []), 0, ""


def _sqlcipher_query(db_path: str, query: str, config: dict, verbose: bool = True) -> pd.DataFrame:
    df, _, _ = _sqlcipher_query_status(db_path, query, config, verbose=verbose)
    return df


def _sqlcipher_paginated_query(db_path: str, query: str, config: dict, page_size: int = 5000) -> pd.DataFrame:
    """全量查询失败时，用分页逐批读取，遇到损坏页则停止并返回已读取数据。"""
    dfs = []
    offset = 0
    while True:
        paged_q = f"{query} LIMIT {page_size} OFFSET {offset}"
        df = _sqlcipher_query(db_path, paged_q, config, verbose=False)
        if df.empty:
            break
        dfs.append(df)
        if len(df) < page_size:
            break
        offset += page_size
    if not dfs:
        return pd.DataFrame()
    result = pd.concat(dfs, ignore_index=True)
    logger.warning("数据库部分损坏，已提取 %d 条（后续数据因 rc=11 跳过）", len(result))
    return result


def _build_sqlcipher_group_chunk_query(identifier: int, after_row_id: int, limit: int) -> str:
    return f"""
        SELECT
            "40001" AS row_id,
            "40033" AS sender_id,
            "40090" AS group_nickname,
            "40093" AS qq_name,
            CAST("40800" AS TEXT) AS content,
            "40050" AS timestamp
        FROM group_msg_table
        WHERE "40027" = {identifier}
          AND "40011" = 2
          AND "40012" = 1
          AND "40800" IS NOT NULL
          AND length("40800") > 0
          AND "40001" > {after_row_id}
        ORDER BY "40001"
        LIMIT {limit}
    """


def _sqlcipher_find_next_readable_group_chunk(db_path: str, identifier: int, config: dict, after_row_id: int,
                                              page_size: int) -> pd.DataFrame:
    probe_steps = [0]
    for exp in range(0, 19):
        base = 10 ** exp
        probe_steps.extend([base, 2 * base, 5 * base])
    seen_probe_ids = set()
    for step in probe_steps:
        probe_row_id = after_row_id + step
        if probe_row_id in seen_probe_ids:
            continue
        seen_probe_ids.add(probe_row_id)
        probe_query = _build_sqlcipher_group_chunk_query(identifier, probe_row_id, min(page_size, 1000))
        df, rc, _ = _sqlcipher_query_status(db_path, probe_query, config, verbose=False)
        if rc == 0 and not df.empty:
            return df
    return pd.DataFrame()


def _sqlcipher_resilient_group_query(db_path: str, identifier: int, config: dict, page_size: int = 5000) -> pd.DataFrame:
    dfs = []
    after_row_id = 0
    last_good_ts = None
    skipped_ranges = []

    while True:
        chunk_query = _build_sqlcipher_group_chunk_query(identifier, after_row_id, page_size)
        df, rc, _ = _sqlcipher_query_status(db_path, chunk_query, config, verbose=False)
        if rc == 0:
            if df.empty:
                break
            dfs.append(df)
            after_row_id = int(df["row_id"].iloc[-1])
            last_good_ts = df["timestamp"].iloc[-1]
            if len(df) < page_size:
                break
            continue

        if last_good_ts is None:
            logger.error("SQLCipher 容错提取失败：首批数据即不可读。")
            return pd.DataFrame()

        recovery_df = _sqlcipher_find_next_readable_group_chunk(db_path, identifier, config, after_row_id, page_size)
        if recovery_df.empty:
            skipped_ranges.append((last_good_ts, None))
            logger.warning(
                "检测到数据库尾部存在不可读页，已跳过：%s 之后的部分消息不可读。",
                _format_unix_timestamp_local(last_good_ts),
            )
            break

        recovered_start_ts = recovery_df["timestamp"].iloc[0]
        skipped_ranges.append((last_good_ts, recovered_start_ts))
        logger.warning(
            "检测到数据库坏页，已跳过不可读区段：%s - %s。",
            _format_unix_timestamp_local(last_good_ts),
            _format_unix_timestamp_local(recovered_start_ts),
        )
        dfs.append(recovery_df)
        after_row_id = int(recovery_df["row_id"].iloc[-1])
        last_good_ts = recovery_df["timestamp"].iloc[-1]

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    result = result.drop_duplicates(subset=["row_id"], keep="first")
    logger.warning(
        "数据库存在坏页，但已尽量恢复可读数据：共提取 %d 条，跳过 %d 个不可读区段。",
        len(result),
        len(skipped_ranges),
    )
    return result


def _build_local_query(mode: str, identifier: int, encrypted: bool, c2c_peer_ids=None) -> tuple:
    """构建本地 SQLite 查询语句。返回 (query, params) 元组。"""
    if mode == "group":
        table = "group_msg_table"
        param = str(identifier) if encrypted else "?"
        where_clause = f'"40027" = {param}'
        params = () if encrypted else (identifier,)
        query = f"""
            SELECT
                "40033" AS sender_id,
                "40090" AS group_nickname,
                "40093" AS qq_name,
                CAST("40800" AS TEXT) AS content,
                "40050" AS timestamp
            FROM {table}
            WHERE {where_clause}
              AND "40011" = 2
              AND "40012" = 1
              AND "40800" IS NOT NULL
              AND length("40800") > 0
        """
        return query, params

    if not c2c_peer_ids:
        raise ValueError("c2c 模式缺少私聊对象列表（40030）。")

    if encrypted:
        self_param = str(identifier)
        peer_filter = ", ".join(c2c_peer_ids)
        params = ()
    else:
        self_param = "?"
        peer_filter = ", ".join(["?"] * len(c2c_peer_ids))
        params = tuple(c2c_peer_ids) + (identifier,)

    query = f"""
        SELECT
            "40033" AS sender_id,
            "40030" AS peer_id,
            CASE
                WHEN "40033" = {self_param} THEN '我'
                ELSE COALESCE(NULLIF(TRIM("40093"), ''), CAST("40033" AS TEXT))
            END AS qq_name,
            CAST("40800" AS TEXT) AS content,
            "40050" AS timestamp
        FROM c2c_msg_table
        WHERE "40030" IN ({peer_filter})
          AND "40011" = 2
          AND "40012" = 1
          AND "40800" IS NOT NULL
          AND length("40800") > 0
        ORDER BY "40050"
    """
    return query, params


def _postprocess(df: pd.DataFrame, mode: str, identifier: int) -> pd.DataFrame:
    """时间戳转换、昵称合并、内容清洗。"""
    if df.empty:
        logger.info("%s 模式下，标识符 %s 未提取到有效数据。", mode, identifier)
        return df

    logger.info("原始时间戳数据：%s", df['timestamp'].head().tolist())
    df['sender_id'] = df['sender_id'].astype(str)
    try:
        df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp'], errors='coerce'), unit='s', errors='coerce', utc=True)
        converted = df['timestamp'].dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
        if not converted.isna().all():
            df['timestamp'] = converted
        else:
            logger.warning("时间转换失败，保留 UTC 时间")
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    except Exception as e:
        logger.warning("时间戳转换失败: %s", e)
    df = df[~df['sender_id'].astype(str).isin(['2854196310', '10000'])]
    df['content'] = df['content'].apply(clean_message)
    if 'group_nickname' in df.columns:
        df['sender_nickname'] = df['group_nickname'].fillna('').str.strip()
        mask = df['sender_nickname'] == ''
        df.loc[mask, 'sender_nickname'] = df.loc[mask, 'qq_name']
        df.drop(columns=['group_nickname', 'qq_name'], inplace=True)
    else:
        df['sender_nickname'] = df['qq_name'].fillna('').astype(str).str.strip()
        mask = df['sender_nickname'] == ''
        df.loc[mask, 'sender_nickname'] = df.loc[mask, 'sender_id'].astype(str)
        df.drop(columns=['qq_name'], inplace=True)
    preferred_names = {}
    for sender_id, group in df.groupby('sender_id'):
        candidates = [
            name for name in group['sender_nickname'].fillna('').astype(str)
            if name and name != sender_id and not name.isdigit() and name.lower() != 'nan'
        ]
        if candidates:
            preferred_names[str(sender_id)] = max(candidates, key=len)
    if preferred_names:
        df['sender_nickname'] = df.apply(
            lambda row: preferred_names.get(
                str(row['sender_id']),
                row['sender_nickname']
            ) if (
                not str(row['sender_nickname']).strip()
                or str(row['sender_nickname']).isdigit()
                or str(row['sender_nickname']).lower() == 'nan'
            ) else row['sender_nickname'],
            axis=1
        )
    return df


def extract_chat_data(db_path: str, identifier: int, mode: str = "group",
                      remote: bool = False, cipher_config: dict = None) -> pd.DataFrame:
    if mode not in ("group", "c2c"):
        logger.error("未知的 mode: %s", mode)
        return pd.DataFrame()

    # 远程 PostgreSQL
    if remote:
        from sqlalchemy import create_engine
        try:
            engine = create_engine(db_path)
        except Exception as e:
            logger.error("无法连接远程数据库: %s", e)
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
                  AND ((message->>'sender_id')::bigint = %s OR (message->>'receiver_id')::bigint = %s)
                  AND (message->>'sender_id')::bigint NOT IN (2854196310, 10000)
                ORDER BY time
            """
        pg_params = (identifier,) if mode == "group" else (identifier, identifier)
        try:
            df = pd.read_sql_query(query, engine, params=pg_params)
        except Exception as e:
            logger.error("执行 SQL 查询失败: %s", e)
            return pd.DataFrame()
        return df

    # 加密本地 SQLite
    config = cipher_config if cipher_config is not None else _load_config()
    if config.get('encrypted', False):
        c2c_peer_ids = None
        if mode == "c2c":
            c2c_peer_ids = _resolve_local_c2c_peer_ids_sqlcipher(db_path, identifier, config)
            if not c2c_peer_ids:
                logger.error("未找到 QQ %s 对应的本地私聊对象列表（40030）。", identifier)
                return pd.DataFrame()
        query, _ = _build_local_query(mode, identifier, encrypted=True, c2c_peer_ids=c2c_peer_ids)
        df = _sqlcipher_query(db_path, query, config)
        if df.empty and mode == "group":
            df = _sqlcipher_resilient_group_query(db_path, identifier, config)
        return _postprocess(df, mode, identifier)

    # 明文本地 SQLite
    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        logger.error("无法连接数据库: %s", e)
        return pd.DataFrame()
    c2c_peer_ids = None
    if mode == "c2c":
        c2c_peer_ids = _resolve_local_c2c_peer_ids_sqlite(conn, identifier)
        if not c2c_peer_ids:
            logger.error("未找到 QQ %s 对应的本地私聊对象列表（40030）。", identifier)
            return pd.DataFrame()
    query, params = _build_local_query(mode, identifier, encrypted=False, c2c_peer_ids=c2c_peer_ids)
    try:
        df = pd.read_sql_query(query, conn, params=params if params else None)
    except Exception as e:
        logger.error("执行 SQL 查询失败: %s", e)
        return pd.DataFrame()
    finally:
        conn.close()
    return _postprocess(df, mode, identifier)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = extract_chat_data("nt_msg.clean.db", 98765432, mode="group")
    logger.info("群聊模式下提取到 %d 条消息记录", len(data))
