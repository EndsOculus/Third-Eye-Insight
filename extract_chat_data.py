"""
extract_chat_data.py
--------------------
从未加密的 SQLite 数据库中提取聊天数据，并清洗掉除 emoji、中文、英文、数字、常见符号以外的其他字符（将无法显示的字符替换为 '?'）。
要求：
- 对于群聊模式，从表 group_msg_table 中提取数据；
- 对于私聊模式，从表 c2c_msg_table 中提取数据；
- 自动过滤 QQ 号为 10000 和 2854196310 的系统消息；
- 同时提取群昵称（40090）和 QQ 名称（40093），默认使用群昵称，若为空则使用 QQ 名称；
- 只提取文本消息（例如 40011 = 2 且 40012 = 1）。
"""

import sqlite3
import pandas as pd

def clean_message(text: str) -> str:
    """
    清洗消息内容：将文本转换为 GBK 编码，无法编码的字符替换为 '?'。
    """
    try:
        return text.encode('gbk', errors='replace').decode('gbk')
    except Exception:
        return text

def extract_chat_data(db_path: str, identifier: int, mode: str = "group") -> pd.DataFrame:
    """
    从数据库中提取聊天数据，并对消息内容进行清洗。
    
    参数：
        db_path (str): 数据库文件路径，例如 "nt_msg.clean.db"。
        identifier (int): 若 mode 为 "group"，则为群聊号码；若为 "c2c"，则为好友 QQ 号。
        mode (str): "group" 表示群聊模式，"c2c" 表示私聊模式。
    
    返回：
        DataFrame，包含以下字段：
          - sender_id: 发送者 QQ 号（字符串类型）
          - sender_nickname: 显示名称（优先使用群昵称，若为空则使用 QQ 名称）
          - content: 消息内容（文本，已清洗）
          - timestamp: 消息发送时间（转换为 datetime 格式，按北京时间）
    """
    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        print(f"[ERROR] 无法连接数据库: {e}")
        return pd.DataFrame()

    if mode == "group":
        query = """
        SELECT 
            "40033" AS sender_id,
            "40090" AS group_nickname,
            "40093" AS qq_name,
            "40080" AS content,
            "40050" AS timestamp
        FROM group_msg_table
        WHERE "40027" = ? 
          AND "40011" = 2 
          AND "40012" = 1 
          AND content IS NOT NULL 
          AND TRIM(content) <> ''
          AND "40033" NOT IN (2854196310, 10000)
        """
    elif mode == "c2c":
        query = """
        SELECT 
            "40033" AS sender_id,
            "40090" AS group_nickname,
            "40093" AS qq_name,
            "40080" AS content,
            "40050" AS timestamp
        FROM c2c_msg_table
        WHERE "40033" = ? 
          AND "40011" = 2 
          AND "40012" = 1 
          AND content IS NOT NULL 
          AND TRIM(content) <> ''
          AND "40033" NOT IN (2854196310, 10000)
        """
    else:
        print(f"[ERROR] 未知的 mode: {mode}")
        conn.close()
        return pd.DataFrame()

    try:
        df = pd.read_sql_query(query, conn, params=(identifier,))
    except Exception as e:
        print(f"[ERROR] 执行 SQL 查询失败: {e}")
        conn.close()
        return pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        print(f"[INFO] {mode} 模式下，标识符 {identifier} 未提取到有效数据。")
        return df
    print("原始时间戳数据：", df['timestamp'].head())

    df['sender_id'] = df['sender_id'].astype(str)
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce', utc=True)
        converted = df['timestamp'].dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
        if not converted.isna().all():
            df['timestamp'] = converted
        else:
            print("[WARN] 时间转换失败，保留 UTC 时间")
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    except Exception as e:
        print(f"[WARN] 时间戳转换失败: {e}")
    
    # 清洗消息内容
    df['content'] = df['content'].apply(clean_message)
    
    # 合并昵称：优先使用 group_nickname，若为空则使用 qq_name
    df['sender_nickname'] = df['group_nickname'].fillna('').str.strip()
    mask = df['sender_nickname'] == ''
    df.loc[mask, 'sender_nickname'] = df.loc[mask, 'qq_name']
    df.drop(columns=['group_nickname', 'qq_name'], inplace=True)

    return df

if __name__ == "__main__":
    db_file = "nt_msg.clean.db"
    data = extract_chat_data(db_file, 98765432, mode="group")
    print(f"群聊模式下提取到 {len(data)} 条消息记录")
