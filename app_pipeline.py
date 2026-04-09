"""
app_pipeline.py
---------------
编排模块：协调数据提取、特征计算、可视化、报告生成的主流程。
"""

import json
import logging
import os
import warnings
from collections import Counter
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from analysis_types import AnalysisParams
from embedding_cache import DEFAULT_MODEL_NAME, attach_text_embeddings
from extract_chat_data import build_display_name_map, extract_chat_data
from metrics import compute_metrics, optimize_weights
from reporting import maybe_generate_report, write_scores_csv
from runtime_config import (
    build_preferred_display_names,
    interactive_config,
    load_analysis_params,
    load_cipher_config,
    maybe_strip_nt_msg_header,
    resolve_device,
)
from visualization import (
    filter_for_gbk,
    plot_custom_heatmap,
    plot_focus_row_heatmap,
    plot_interaction_network,
    plot_top_pairs,
)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
warnings.filterwarnings("ignore", message="You are sending unauthenticated requests to the HF Hub.*")

logger = logging.getLogger(__name__)



def build_output_dir(mode, identifier):
    now = datetime.now()
    mode_dir = "group" if mode == "group" else "private"
    return os.path.join("output", now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"), mode_dir, str(identifier))


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


def write_user_mapping(out_dir, users, user_name_map):
    mapping_path = os.path.join(out_dir, "user_mapping.txt")
    with open(mapping_path, "w", encoding="utf-8-sig") as handle:
        handle.write("索引\tQQ号\t昵称\n")
        for idx, user_id in enumerate(users):
            handle.write(f"{idx}\t{user_id}\t{user_name_map.get(user_id, str(user_id))}\n")
    logger.info(f"用户映射文件已保存到 {mapping_path}")


def compute_final_intimacy(args, metrics):
    if args.mode == "c2c":
        weights = (0.4, 0.4, 0.2)
        logger.info(f"私聊模式使用固定权重：语义上下文 {weights[0]:.2f}, 行为特征 {weights[1]:.2f}, 时间粘性 {weights[2]:.2f}")
    elif args.auto_weight:
        weights = tuple(optimize_weights(metrics.semantic_matrix_mapped, metrics.behavior_norm, metrics.third_matrix))
        logger.info(f"自动调整权重：语义 {weights[0]:.2f}, 行为 {weights[1]:.2f}, {metrics.third_metric_name} {weights[2]:.2f}")
    else:
        weights = (0.4, 0.4, 0.2)
    final_intimacy = np.maximum(
        weights[0] * metrics.semantic_matrix_mapped + weights[1] * metrics.behavior_norm + weights[2] * metrics.third_matrix,
        0,
    )
    logger.info(f"最终互动活跃度得分范围：{final_intimacy.min():.4f} - {final_intimacy.max():.4f}")
    return final_intimacy, weights


def render_visualizations(args, identifier, out_dir, time_range_str, users, user_to_index, labels, user_name_map, focus_user, final_intimacy, metrics, weights):
    num_users = len(users)
    if num_users >= 2:
        focus_target = focus_user if focus_user else (str(identifier) if args.mode == "c2c" else None)
        if focus_target and focus_target in user_to_index:
            focus_idx = user_to_index[focus_target]
            plot_focus_row_heatmap(metrics.semantic_matrix_mapped, labels, focus_idx, title=metrics.semantic_title + time_range_str, save_path=f"{out_dir}/semantic_heatmap{time_range_str}.png")
            plot_focus_row_heatmap(metrics.behavior_norm, labels, focus_idx, title=metrics.behavior_title + time_range_str, save_path=f"{out_dir}/behavior_heatmap{time_range_str}.png")
            plot_focus_row_heatmap(metrics.third_matrix, labels, focus_idx, title=metrics.third_title + time_range_str, save_path=f"{out_dir}/network_heatmap{time_range_str}.png")
            plot_focus_row_heatmap(final_intimacy, labels, focus_idx, title="综合亲密度热力图" + time_range_str, save_path=f"{out_dir}/intimacy_heatmap{time_range_str}.png")
        else:
            plot_custom_heatmap(metrics.semantic_matrix_mapped, labels, title=metrics.semantic_title + time_range_str, save_path=f"{out_dir}/semantic_heatmap{time_range_str}.png")
            plot_custom_heatmap(metrics.behavior_norm, labels, title=metrics.behavior_title + time_range_str, save_path=f"{out_dir}/behavior_heatmap{time_range_str}.png")
            plot_custom_heatmap(metrics.third_matrix, labels, title=metrics.third_title + time_range_str, save_path=f"{out_dir}/network_heatmap{time_range_str}.png")
            plot_custom_heatmap(final_intimacy, labels, title="综合亲密度热力图" + time_range_str, save_path=f"{out_dir}/intimacy_heatmap{time_range_str}.png")
        plot_top_pairs(
            final_intimacy,
            metrics.behavior_norm,
            metrics.semantic_matrix_mapped,
            metrics.third_matrix,
            users,
            user_name_map,
            save_path=f"{out_dir}/top_pairs{time_range_str}.png",
            third_label=metrics.third_metric_name,
            component_weights=weights,
            focus_user=focus_user,
        )
    else:
        logger.warning("用户数不足 2，跳过热力图和 Top 互动对图。")

    edges = [
        (i, j, (metrics.third_matrix[i, j] + metrics.semantic_matrix_mapped[i, j] + metrics.behavior_norm[i, j]) / 3)
        for i in range(num_users)
        for j in range(i + 1, num_users)
        if metrics.interaction_weights[i, j] > 0
    ]
    if focus_user:
        focus_indices = {idx for idx, user in enumerate(users) if user == focus_user}
        edges = [(i, j, w) for i, j, w in edges if i in focus_indices or j in focus_indices]
    if edges:
        plot_interaction_network(edges, labels, save_path=f"{out_dir}/interaction_network{time_range_str}.png")
    else:
        logger.warning("没有可绘制的互动边，跳过网络图生成。")


def run_analysis():
    logging.basicConfig(level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO), format="%(levelname)s %(message)s")

    cipher_config = {}
    if os.path.exists("config.json"):
        with open("config.json", "r", encoding="utf-8") as handle:
            cipher_config = json.load(handle)
    analysis_params = load_analysis_params(cipher_config)

    maybe_strip_nt_msg_header()
    args = interactive_config(cipher_config)
    device = resolve_device(args.boost)

    focus_user = str(args.focus_user) if args.focus_user else None
    identifier = args.group if args.mode == "group" else args.id
    if args.mode == "c2c":
        focus_user = str(identifier)

    out_dir = build_output_dir(args.mode, identifier)
    os.makedirs(out_dir, exist_ok=True)

    logger.info("正在提取数据...")
    chat_df = extract_chat_data(args.db, identifier, mode=args.mode, remote=args.remote, cipher_config=cipher_config)
    if chat_df.empty:
        logger.error("未提取到数据，程序退出。")
        raise SystemExit(1)
    logger.info(f"提取到 {len(chat_df)} 条消息记录。")

    preferred_display_names = build_preferred_display_names(chat_df, args, identifier, cipher_config)
    chat_df, time_range_str = prompt_and_apply_time_filters(chat_df)
    chat_df = apply_exclude_users(chat_df, args.exclude_users)
    chat_df = chat_df.sort_values("timestamp").reset_index(drop=True)

    if args.lite and focus_user:
        chat_df = apply_lite_filter(chat_df, focus_user, analysis_params.WINDOW_SIZE)

    logger.info("加载预训练文本嵌入模型并进行批量计算...")
    chat_df = attach_text_embeddings(
        chat_df,
        DEFAULT_MODEL_NAME,
        batch_size=analysis_params.BATCH_SIZE,
        num_workers=analysis_params.N_THREADS,
        device=device,
    )
    logger.info("文本嵌入计算完成。")

    users, user_to_index = build_user_index(chat_df)
    if len(users) < 2:
        logger.warning(f"当前筛选后的数据仅包含 {len(users)} 个用户，无法形成用户对互动矩阵。")
        logger.warning("这通常说明私聊提取逻辑没有正确区分自己和对方，或当前时间范围内只剩单方消息。")

    metrics = compute_metrics(chat_df, args, identifier, users, user_to_index, analysis_params)
    if args.mode == "c2c":
        logger.debug(f"行为特征得分范围：{metrics.behavior_norm.min():.4f} - {metrics.behavior_norm.max():.4f}")
        logger.debug(f"语义上下文得分（前5x5）：\n{metrics.semantic_matrix_mapped[:5, :5]}")
        logger.debug(f"时间粘性得分范围：{metrics.third_matrix.min():.4f} - {metrics.third_matrix.max():.4f}")

    user_name_map = build_user_name_map(chat_df, users, preferred_display_names)
    labels = [filter_for_gbk(user_name_map.get(user, str(user))) for user in users]
    write_user_mapping(out_dir, users, user_name_map)

    final_intimacy, weights = compute_final_intimacy(args, metrics)
    render_visualizations(args, identifier, out_dir, time_range_str, users, user_to_index, labels, user_name_map, focus_user, final_intimacy, metrics, weights)
    write_scores_csv(out_dir, time_range_str, args, users, user_name_map, metrics, final_intimacy, focus_user)
    maybe_generate_report(args, out_dir, chat_df, users, user_name_map, metrics, final_intimacy, weights, analysis_params.TOP_N, focus_user)

    logger.info(f"分析完成。结果已保存至 {out_dir}/")


if __name__ == "__main__":
    run_analysis()
