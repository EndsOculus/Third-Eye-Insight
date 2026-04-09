import json
import logging
import os
from collections import Counter
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import torch

from analysis_types import AnalysisParams
from extract_chat_data import build_display_name_map

logger = logging.getLogger(__name__)


def ask(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else default


def ask_yn(prompt, default=False):
    hint = "[Y/n]" if default else "[y/N]"
    val = input(f"{prompt} {hint} ").strip().lower()
    if not val:
        return default
    return val == "y"


def parse_exclude_users(raw_value):
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def load_analysis_params(cipher_config):
    raw = cipher_config.get("analysis_params", {})
    return AnalysisParams(
        L=int(raw.get("L", 1000)),
        TAU=int(raw.get("TAU", 150)),
        WINDOW_SIZE=int(raw.get("WINDOW_SIZE", 300)),
        BATCH_SIZE=int(raw.get("BATCH_SIZE", 32)),
        N_THREADS=int(raw.get("N_THREADS", 4)),
        TOP_N=int(raw.get("TOP_N", 20)),
    )


def load_cipher_config():
    if os.path.exists("config.json"):
        with open("config.json", "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def interactive_config(cipher_config: dict) -> SimpleNamespace:
    logger.info("======== 恋之瞳 · 群聊互动分析 ========")
    default_db = cipher_config.get("db_file", "nt_msg.clean.db")
    db = ask("数据库文件路径", default_db)

    logger.info("分析模式：")
    logger.info("  1. 群聊")
    logger.info("  2. 私聊")
    mode_choice = ask("请选择", "1")
    mode = "c2c" if mode_choice.strip() == "2" else "group"

    if mode == "group":
        while True:
            raw = ask("群号")
            if raw and raw.isdigit():
                group = int(raw)
                break
            logger.warning("请输入有效的群号。")
        identifier = group
    else:
        group = 0
        while True:
            raw = ask("我的 QQ 号（将分析与所有私聊好友的互动）")
            if raw and raw.isdigit():
                identifier = int(raw)
                break
            logger.warning("请输入有效的 QQ 号。")

    focus_user = None
    lite = False
    if mode == "group":
        focus_raw = ask("聚焦分析某用户的 QQ 号（回车跳过）", "")
        focus_user = focus_raw if focus_raw else None
        if focus_user:
            lite = ask_yn("  启用精简模式（仅保留该用户相关互动）")

    exclude_raw = ask("排除用户 QQ 号（多个用逗号分隔，回车跳过）", "")
    remote = ask_yn("使用远程数据库连接")
    boost = ask_yn("启用 GPU 加速")
    auto_wt = ask_yn("自动调整融合权重")
    report = ask_yn("生成 AI 分析报告（需 DEEPSEEK_API_KEY 环境变量）")
    font = ask("图表中文字体", "Microsoft YaHei")

    return SimpleNamespace(
        db=db,
        mode=mode,
        group=group,
        id=identifier,
        focus_user=focus_user,
        lite=lite,
        exclude_users=parse_exclude_users(exclude_raw),
        remote=remote,
        boost=boost,
        auto_weight=auto_wt,
        report=report,
        font=font,
    )


def maybe_strip_nt_msg_header():
    if not os.path.exists("nt_msg.db"):
        return
    if ask_yn("发现 nt_msg.db，是否剥离 1024 字节文件头生成 nt_msg.clean_e.db"):
        with open("nt_msg.db", "rb") as fin:
            fin.seek(1024)
            stripped = fin.read()
        with open("nt_msg.clean_e.db", "wb") as fout:
            fout.write(stripped)
        logger.info("nt_msg.clean_e.db 已生成。")
        if ask_yn("是否删除 nt_msg.db"):
            os.remove("nt_msg.db")
            logger.info("nt_msg.db 已删除。")


def resolve_device(use_gpu):
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"使用设备：cuda ({torch.cuda.get_device_name(0)})")
            return device
        logger.warning("启用 GPU 加速失败，回退 CPU。")
        logger.info(f"torch.__version__     = {torch.__version__}")
        logger.info(f"torch.version.cuda    = {torch.version.cuda}")
        logger.info("cuda.is_available()   = False")
    device = torch.device("cpu")
    logger.info("使用设备：cpu")
    return device


def build_output_dir(mode, identifier):
    now = datetime.now()
    mode_dir = "group" if mode == "group" else "private"
    return os.path.join("output", now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"), mode_dir, str(identifier))


def build_preferred_display_names(chat_df, args, identifier, cipher_config):
    preferred_display_names = {}
    for sender_id, group in chat_df.groupby("sender_id"):
        candidates = []
        for raw_name in group["sender_nickname"].astype(str):
            name = raw_name.strip()
            if not name or name.lower() == "nan" or name.isdigit():
                continue
            candidates.append(name)
        if candidates:
            ranked = sorted(Counter(candidates).items(), key=lambda item: (item[1], len(item[0])), reverse=True)
            preferred_display_names[str(sender_id)] = ranked[0][0]

    preferred_display_names[str(identifier)] = "我" if args.mode == "c2c" else preferred_display_names.get(str(identifier), str(identifier))
    missing_name_ids = [
        sender_id
        for sender_id in chat_df["sender_id"].astype(str).unique()
        if sender_id not in preferred_display_names or preferred_display_names.get(sender_id, "").isdigit()
    ]
    alias_overrides = build_display_name_map(args.db, missing_name_ids, remote=args.remote, cipher_config=cipher_config)
    preferred_display_names.update(alias_overrides)
    preferred_display_names[str(identifier)] = "我" if args.mode == "c2c" else preferred_display_names.get(str(identifier), str(identifier))
    return preferred_display_names


def prompt_and_apply_time_filters(chat_df):
    logger.info("正在清洗数据...")
    logger.info(f"数据时间范围：{chat_df['timestamp'].min()} 到 {chat_df['timestamp'].max()}")
    start_date_str = input("请输入起始日期（例如 2024/01/01），直接回车表示不限制：").strip()
    end_date_str = input("请输入结束日期（例如 2024/12/31），直接回车表示不限制：").strip()

    if start_date_str:
        try:
            start_date = datetime.strptime(start_date_str, "%Y/%m/%d")
            chat_df = chat_df[chat_df["timestamp"] >= start_date]
            logger.info(f"筛选后起始日期：{start_date.date()}，剩余 {len(chat_df)} 条记录。")
        except Exception as exc:
            logger.error(f"起始日期解析失败：{exc}")
            raise SystemExit(1) from exc

    if end_date_str:
        try:
            end_date = datetime.strptime(end_date_str, "%Y/%m/%d")
            chat_df = chat_df[chat_df["timestamp"] <= end_date]
            logger.info(f"筛选后截止日期：{end_date.date()}，剩余 {len(chat_df)} 条记录。")
        except Exception as exc:
            logger.error(f"结束日期解析失败：{exc}")
            raise SystemExit(1) from exc

    if chat_df.empty:
        logger.error("筛选后数据为空，请检查时间范围。")
        raise SystemExit(1)

    time_range_str = ""
    if start_date_str or end_date_str:
        start_part = start_date_str.replace("/", "-") if start_date_str else "start"
        end_part = end_date_str.replace("/", "-") if end_date_str else "end"
        time_range_str = f"_{start_part}-{end_part}"
    return chat_df, time_range_str


def apply_exclude_users(chat_df, exclude_users):
    if not exclude_users:
        return chat_df
    chat_df = chat_df[~chat_df["sender_id"].astype(str).isin(exclude_users)]
    logger.info(f"排除 {exclude_users} 后，剩余 {len(chat_df)} 条记录。")
    if chat_df.empty:
        logger.error("排除后数据为空，请检查 QQ 号是否正确。")
        raise SystemExit(1)
    return chat_df


def apply_lite_filter(chat_df, focus_user, window_size):
    related_indices = set()
    for i in range(len(chat_df) - 1):
        current_row = chat_df.iloc[i]
        next_row = chat_df.iloc[i + 1]
        if (
            current_row["sender_id"] == focus_user or next_row["sender_id"] == focus_user
        ) and (next_row["timestamp"] - current_row["timestamp"]).total_seconds() <= window_size:
            related_indices.add(i)
            related_indices.add(i + 1)
    filtered_df = chat_df.iloc[sorted(related_indices)].reset_index(drop=True)
    logger.info(f"仅保留与用户 {focus_user} 相关的互动记录，共 {len(filtered_df)} 条记录。")
    return filtered_df


def build_user_index(chat_df):
    users = list(chat_df["sender_id"].astype(str).unique())
    user_to_index = {user: idx for idx, user in enumerate(users)}
    return users, user_to_index


def build_user_name_map(chat_df, users, preferred_display_names):
    user_name_map = {}
    for _, row in chat_df.iterrows():
        nickname = row["sender_nickname"]
        if pd.isna(nickname) or str(nickname).strip() == "":
            nickname = str(row["sender_id"])
        sender_id = str(row["sender_id"])
        user_name_map[sender_id] = preferred_display_names.get(sender_id, str(nickname))
    for user_id in users:
        user_name_map.setdefault(str(user_id), preferred_display_names.get(str(user_id), str(user_id)))
    return user_name_map
