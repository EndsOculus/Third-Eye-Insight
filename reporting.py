"""Reporting module for CSV export and AI report generation."""

import csv
import logging
import os
from datetime import datetime

from visualization import filter_for_gbk

logger = logging.getLogger(__name__)


def build_pair_indices(users, focus_user):
    if focus_user:
        return [(i, j) for i in range(len(users)) for j in range(i + 1, len(users)) if users[i] == focus_user or users[j] == focus_user]
    return [(i, j) for i in range(len(users)) for j in range(i + 1, len(users))]


def write_scores_csv(out_dir, time_range_str, args, users, user_name_map, metrics, final_intimacy, focus_user):
    pair_indices = build_pair_indices(users, focus_user)
    csv_path = f"{out_dir}/interaction_scores{time_range_str}.csv"
    candidate_paths = [csv_path]
    fallback_suffix = datetime.now().strftime("%H%M%S")
    candidate_paths.append(f"{out_dir}/interaction_scores{time_range_str}_{fallback_suffix}.csv")
    last_error = None

    for candidate_path in candidate_paths:
        try:
            with open(candidate_path, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                third_score_header = "NetworkScore" if args.mode == "group" else "StickinessScore"
                behavior_score_header = "BehaviorScore" if args.mode == "group" else "BehavioralScore"
                semantic_score_header = "SemanticScore" if args.mode == "group" else "SemanticContextScore"
                writer.writerow(["UserID1", "UserName1", "UserID2", "UserName2", behavior_score_header, semantic_score_header, third_score_header, "IntimacyScore"])
                for i, j in pair_indices:
                    writer.writerow(
                        [
                            users[i],
                            filter_for_gbk(user_name_map.get(users[i], str(users[i]))),
                            users[j],
                            filter_for_gbk(user_name_map.get(users[j], str(users[j]))),
                            f"{metrics.behavior_norm[i, j]:.4f}",
                            f"{metrics.semantic_matrix_mapped[i, j]:.4f}",
                            f"{metrics.third_matrix[i, j]:.4f}",
                            f"{final_intimacy[i, j]:.4f}",
                        ]
                    )
            if candidate_path != csv_path:
                logger.warning("目标 CSV 被占用，已改写到备用文件：%s", candidate_path)
            else:
                logger.info(f"CSV 文件已保存到 {candidate_path}")
            return candidate_path
        except PermissionError as exc:
            last_error = exc
            continue

    raise last_error if last_error else PermissionError(f"无法写入 CSV 文件：{csv_path}")


def generate_report_via_api(api_key, report_content, save_path="output/analysis_report.md", system_prompt="You are a helpful assistant."):
    import openai

    try:
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": report_content}],
            stream=False,
        )
        report_text = response.choices[0].message.content
        with open(save_path, "w", encoding="utf-8") as handle:
            handle.write(report_text)
        logger.info(f"分析报告已保存到 {save_path}")
    except Exception as exc:
        logger.error(f"调用 API 生成报告时出错: {exc}")


def build_report_prompt(args, chat_df, users, user_name_map, metrics, final_intimacy, weights, top_n, focus_user):
    msg_counts = chat_df["sender_id"].astype(str).value_counts()
    
    # 聚焦模式下，user_table 只显示 focus_user 的互动对象及其互动消息数
    if args.mode == "group" and focus_user:
        focus_user_str = str(focus_user)
        user_table = "## 互动对象列表\n\n| 昵称 | QQ号 | 与焦点用户的互动消息数 |\n| --- | --- | --- |\n"
        focus_idx = users.index(focus_user_str) if focus_user_str in users else -1
        if focus_idx >= 0:
            interaction_scores = []
            for i, user in enumerate(users):
                if i != focus_idx:
                    intimacy = final_intimacy[focus_idx, i] if focus_idx < i else final_intimacy[i, focus_idx]
                    interaction_scores.append((user, intimacy, msg_counts.get(user, 0)))
            interaction_scores.sort(key=lambda x: x[1], reverse=True)
            for user, intimacy, msg_cnt in interaction_scores:
                user_table += f"| {user_name_map.get(user, user)} | {user} | ~{max(1, int(msg_cnt * intimacy / 100))} |\n"
    else:
        user_table = "## 用户列表\n\n| 昵称 | QQ号 | 消息数 |\n| --- | --- | --- |\n"
        for user in users:
            user_table += f"| {user_name_map.get(user, user)} | {user} | {msg_counts.get(user, 0)} |\n"

    pair_scores = []
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            if final_intimacy[i, j] > 0:
                # 聚焦模式下，只包含 focus_user 参与的对
                if args.mode == "group" and focus_user:
                    focus_user_str = str(focus_user)
                    if users[i] != focus_user_str and users[j] != focus_user_str:
                        continue
                pair_scores.append(
                    (
                        user_name_map.get(users[i], users[i]),
                        user_name_map.get(users[j], users[j]),
                        final_intimacy[i, j],
                        metrics.behavior_norm[i, j],
                        metrics.semantic_matrix_mapped[i, j],
                        metrics.third_matrix[i, j],
                    )
                )
    pair_scores.sort(key=lambda item: item[2], reverse=True)

    third_table_label = "网络" if args.mode == "group" else "时间粘性"
    top_table = f"## Top {top_n} 互动对\n\n| 用户A | 用户B | 亲密度 | 行为 | 语义 | {third_table_label} |\n| --- | --- | --- | --- | --- | --- |\n"
    for name_a, name_b, intimacy, behavior, semantic, third_score in pair_scores[:top_n]:
        top_table += f"| {name_a} | {name_b} | {intimacy:.3f} | {behavior:.3f} | {semantic:.3f} | {third_score:.3f} |\n"

    if args.mode == "group":
        if focus_user:
            scope_desc = f"重点分析用户 {user_name_map.get(focus_user, focus_user)} 在群内的角色与互动特征"
            system_prompt = (
                "你是一名互动网络分析师，专门分析单个用户在群体社交网络中的位置、关系模式与交互风格。"
                "【重要】禁止进行绝对活跃度比较（如'是群里最活跃的人'）。"
                "分析重点：该用户与其他人的互动方式、相对关系强度（只在该用户的朋友圈内排序）、是否是连接者、有无固定小圈子。"
                "假设该用户就是你的分析对象，而非群体俯视地位。"
                "报告使用 Markdown 格式，包含标题、分节和重点加粗。"
            )
            report_requirements = """## 报告要求
1. **互动对象分布**：该用户与哪些人关系密切，与谁的互动相对较弱，以及互动方式的总体特征
2. **Top 互动关系**：该用户的 Top 5 互动伙伴逐一解读（不涉及此人与他人的绝对活跃度对比，只分析关系特征）
3. **人际网络位置**：该用户是否扮演连接者角色？有无特定的小圈子或社交圈层？
4. **互动风格**：该用户vs其各伙伴的语义一致性如何？行为互动是否对称？
5. **角色推断**：基于互动模式而非活跃度绝对值，推断该用户的社交角色定位
"""
        else:
            scope_desc = "群内所有成员互动"
            system_prompt = (
                "你是一名群体社会关系分析师，擅长从聊天数据的量化指标中解读人际关系模式。"
                "请用流畅、洞察深刻的中文撰写分析报告，避免机械罗列数字，重点挖掘有价值的社群结构和人际关系特征。"
                "报告使用 Markdown 格式，包含标题、分节和重点加粗。"
            )
            report_requirements = """## 报告要求
1. **核心关系识别**：Top 5 关系对逐一解读（三维度对比，关系性质判断）
2. **社群圈层分析**：识别群内存在哪些小圈子，各圈子的特征
3. **关键节点**：谁是连接不同圈子的桥梁？谁是信息传播中心？
4. **边缘成员**：哪些用户互动较少？可能原因？
5. **整体评估**：群体活跃度、社群健康度的综合判断和建议
"""
        analysis_object = f"- 群号：{args.group}"
        analysis_title = "QQ 群聊互动分析数据"
        metric_desc = "- **网络得分**：Jaccard 共同邻居系数，反映两人社交圈的重叠程度"
    else:
        scope_desc = f"账号 {args.id} 与所有私聊对象的互动"
        system_prompt = (
            "你是一名一对一关系分析师，擅长从私聊数据的量化指标中解读亲密度、互动节奏与关系变化。"
            "请用流畅、洞察深刻的中文撰写分析报告，避免把私聊误写成群聊或社群分析。"
            "报告使用 Markdown 格式，包含标题、分节和重点加粗。"
        )
        report_requirements = """## 报告要求
1. **核心关系识别**：Top 5 私聊对象逐一解读（三维度对比，关系性质判断）
2. **互动节奏分析**：哪些对象互动更稳定、哪些更像阶段性集中联系
3. **关系层级划分**：区分高频核心联系人、一般联系人、低频联系人
4. **异常与空白**：识别语义强但行为失衡、或时间粘性高但主动性不对称的对象，并解释可能原因
5. **整体评估**：该账号私聊关系结构、活跃度分布与值得关注的联系模式
"""
        analysis_object = f"- 我的 QQ 号：{args.id}"
        analysis_title = "QQ 私聊互动分析数据"
        metric_desc = "- **时间粘性得分**：综合活跃天占比与深夜聊天占比，反映跨时间跨度的联系持续性"

    report_prompt = f"""请根据以下 {analysis_title}，撰写一份结构化的人际关系分析报告。

## 分析背景
{analysis_object}
- 数据范围：{chat_df['timestamp'].min().strftime('%Y/%m/%d')} 至 {chat_df['timestamp'].max().strftime('%Y/%m/%d')}
- 总用户数：{len(users)}，总消息数：{len(chat_df)}
- 分析范围：{scope_desc}
- 融合权重：语义 {weights[0]:.2f} / 行为 {weights[1]:.2f} / {metrics.third_metric_name} {weights[2]:.2f}

## 指标说明
- **行为得分**：综合互动对称性、响应延迟与主动发起平衡度，反映互动是否双向、及时、均衡
- **语义得分**：综合相邻问答语义连贯性与高频词/语气词共现，反映双方是否处于同一语境
{metric_desc}
- **亲密度**：三项加权融合（0-1），综合衡量互动关系强度

{user_table}

{top_table}

{report_requirements}
"""
    return system_prompt, report_prompt


def maybe_generate_report(args, out_dir, chat_df, users, user_name_map, metrics, final_intimacy, weights, top_n, focus_user):
    if not args.report:
        logger.info("未选择生成分析报告，跳过。")
        return
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    system_prompt, report_prompt = build_report_prompt(args, chat_df, users, user_name_map, metrics, final_intimacy, weights, top_n, focus_user)
    generate_report_via_api(api_key, report_prompt, save_path=f"{out_dir}/analysis_report.md", system_prompt=system_prompt)
