"""
app_pipeline.py
---------------
编排模块：协调数据提取、特征计算、可视化、报告生成的主流程。
"""

import logging
import os
import shutil
import subprocess
import warnings
from datetime import datetime

from tqdm import tqdm

from embedding_cache import DEFAULT_MODEL_NAME, attach_text_embeddings
from extract_chat_data import extract_chat_data
from metrics import compute_final_intimacy, compute_metrics
from reporting import maybe_generate_report, write_scores_csv
from runtime_config import (
    apply_exclude_users,
    apply_lite_filter,
    build_output_dir,
    build_preferred_display_names,
    build_user_index,
    build_user_name_map,
    interactive_config,
    load_analysis_params,
    load_cipher_config,
    prompt_and_apply_time_filters,
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


def write_user_mapping(out_dir, users, user_name_map):
    mapping_path = os.path.join(out_dir, "user_mapping.txt")
    with open(mapping_path, "w", encoding="utf-8-sig") as handle:
        handle.write("索引\tQQ号\t昵称\n")
        for idx, user_id in enumerate(users):
            handle.write(f"{idx}\t{user_id}\t{user_name_map.get(user_id, str(user_id))}\n")
    logger.info(f"用户映射文件已保存到 {mapping_path}")


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

    cipher_config = load_cipher_config()
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

    with tqdm(total=8, desc="整体进度", unit="步", ncols=80, leave=True) as pbar:
        pbar.set_postfix_str("数据提取中...")
        logger.info("正在提取数据...")
        chat_df = extract_chat_data(args.db, identifier, mode=args.mode, remote=args.remote, cipher_config=cipher_config)
        if chat_df.empty:
            logger.error("未提取到数据，程序退出。")
            raise SystemExit(1)
        logger.info(f"提取到 {len(chat_df)} 条消息记录。")
        pbar.update(1)

        pbar.set_postfix_str("时间筛选中...")
        preferred_display_names = build_preferred_display_names(chat_df, args, identifier, cipher_config)
        chat_df, time_range_str = prompt_and_apply_time_filters(chat_df)
        pbar.update(1)

        pbar.set_postfix_str("用户过滤中...")
        chat_df = apply_exclude_users(chat_df, args.exclude_users)
        chat_df = chat_df.sort_values("timestamp").reset_index(drop=True)
        if args.lite and focus_user:
            chat_df = apply_lite_filter(chat_df, focus_user, analysis_params.WINDOW_SIZE)
        pbar.update(1)

        pbar.set_postfix_str("embedding编码中...")
        logger.info("加载预训练文本嵌入模型并进行批量计算...")
        chat_df = attach_text_embeddings(
            chat_df,
            DEFAULT_MODEL_NAME,
            batch_size=analysis_params.BATCH_SIZE,
            num_workers=analysis_params.N_THREADS,
            device=device,
        )
        logger.info("文本嵌入计算完成。")
        pbar.update(1)

        pbar.set_postfix_str("指标计算中...")
        users, user_to_index = build_user_index(chat_df)
        if len(users) < 2:
            logger.warning(f"当前筛选后的数据仅包含 {len(users)} 个用户，无法形成用户对互动矩阵。")
            logger.warning("这通常说明私聊提取逻辑没有正确区分自己和对方，或当前时间范围内只剩单方消息。")
        metrics = compute_metrics(chat_df, args, identifier, users, user_to_index, analysis_params)
        if args.mode == "c2c":
            logger.debug(f"行为特征得分范围：{metrics.behavior_norm.min():.4f} - {metrics.behavior_norm.max():.4f}")
            logger.debug(f"语义上下文得分（前5x5）：\n{metrics.semantic_matrix_mapped[:5, :5]}")
            logger.debug(f"时间粘性得分范围：{metrics.third_matrix.min():.4f} - {metrics.third_matrix.max():.4f}")
        pbar.update(1)

        pbar.set_postfix_str("融合权重中...")
        user_name_map = build_user_name_map(chat_df, users, preferred_display_names)
        labels = [filter_for_gbk(user_name_map.get(user, str(user))) for user in users]
        write_user_mapping(out_dir, users, user_name_map)
        final_intimacy, weights = compute_final_intimacy(args, metrics)
        pbar.update(1)

        pbar.set_postfix_str("可视化中...")
        render_visualizations(args, identifier, out_dir, time_range_str, users, user_to_index, labels, user_name_map, focus_user, final_intimacy, metrics, weights)
        pbar.update(1)

        pbar.set_postfix_str("报告生成中...")
        write_scores_csv(out_dir, time_range_str, args, users, user_name_map, metrics, final_intimacy, focus_user)
        maybe_generate_report(args, out_dir, chat_df, users, user_name_map, metrics, final_intimacy, weights, analysis_params.TOP_N, focus_user)
        pbar.update(1)
        pbar.set_postfix_str("完成")

    logger.info(f"分析完成。结果已保存至 {out_dir}/")
    _publish_to_new(out_dir)


def _publish_to_new(out_dir):
    new_dir = os.path.join("output", "new")
    archive_base = os.path.join("output", "archive")
    if os.path.exists(new_dir) and os.listdir(new_dir):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = os.path.join(archive_base, stamp)
        shutil.move(new_dir, dest)
        logger.info(f"旧结果已归档至 {dest}")
    os.makedirs(new_dir, exist_ok=True)
    for fname in os.listdir(out_dir):
        shutil.copy2(os.path.join(out_dir, fname), os.path.join(new_dir, fname))
    logger.info(f"最新结果已复制至 {new_dir}")
    abs_new = os.path.abspath(new_dir)
    try:
        subprocess.Popen(["explorer", abs_new])
    except Exception as exc:
        logger.warning(f"自动打开文件夹失败: {exc}")


if __name__ == "__main__":
    run_analysis()
