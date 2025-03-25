"""
train.py
--------
基于深度学习与文本内容特征融合，分析群聊/私聊中用户互动亲密度。
功能：
1. 命令行参数支持：--group, --db, --mode, --id, --font, --boost（是否启用 GPU 加速，默认关闭），以及 --report（是否生成报告），--focus-user（聚焦某个用户）。
2. 运行时交互式输入起始和结束日期进行时间筛选（格式：YYYY/MM/DD）。
3. 使用 SentenceTransformer 提取文本嵌入，并计算每个用户的平均嵌入（直接均值后归一化）。
4. 构造行为矩阵（统计5分钟内连续消息的互动次数），对 np.log1p(interaction_counts) 结果进行 ECDF 离散化映射。
5. 计算语义相似度矩阵（基于用户平均嵌入计算余弦相似度），对结果采用 ECDF 离散化映射方法拉大微小差异。
6. 融合行为得分和语义得分（各占50%），确保最终得分最低为0。
7. 保存 CSV 文件（GBK编码），只保存唯一用户对（i<j），并支持 --focus-user 参数，仅输出包含指定用户的对。
8. 生成可视化图表（热力图、网络图），以及可选调用 DeepSeek API 自动生成详细分析报告。
"""

import argparse
import numpy as np
import torch
from extract_chat_data import extract_chat_data
from sentence_transformers import SentenceTransformer
from datetime import datetime
import csv
import os
import requests
import math
from sklearn.metrics.pairwise import cosine_similarity

# 导入可视化函数
from visualization import plot_interaction_heatmap, plot_interaction_network, plot_custom_heatmap

# 定义一个离散化映射函数（基于经验累计分布，加入微小扰动以打破完全相同的值）
def discrete_mapping(matrix, L=1000):
    """
    将矩阵中每个值映射到 [0, 1] 的离散级别，共 L 个等级。
    只对上三角数据（唯一用户对）进行处理，加入微小扰动后采用线性插值映射，再离散化。
    """
    triu_indices = np.triu_indices_from(matrix, k=1)
    values = matrix[triu_indices].astype(float)
    jitter = np.random.uniform(-1e-6, 1e-6, size=values.shape)
    values_jitter = values + jitter
    sorted_vals = np.sort(values_jitter)
    mapped_continuous = np.interp(values_jitter, sorted_vals, np.linspace(0, 1, len(sorted_vals)))
    discrete_values = np.round(mapped_continuous * (L - 1)) / (L - 1)
    mapped_matrix = np.copy(matrix)
    for idx, (i, j) in enumerate(zip(triu_indices[0], triu_indices[1])):
        mapped_matrix[i, j] = discrete_values[idx]
        mapped_matrix[j, i] = discrete_values[idx]
    return mapped_matrix

# 解析命令行参数
parser = argparse.ArgumentParser(description="Deep Learning Chat Interaction Affinity Analysis")
parser.add_argument("--group", type=int, required=True, help="Group chat number (effective in group mode)")
parser.add_argument("--db", type=str, required=True, help="Database file path, e.g., nt_msg.clean.db")
parser.add_argument("--mode", type=str, choices=["group", "c2c"], required=True, help="Mode: group for group chat or c2c for private chat")
parser.add_argument("--id", type=int, required=True, help="In group mode, same as --group; in c2c mode, friend QQ number")
parser.add_argument("--font", type=str, default="Microsoft YaHei", help="Font used for charts (default: Microsoft YaHei)")
parser.add_argument("--boost", action="store_true", help="Enable GPU acceleration (default off)")
parser.add_argument("--report", action="store_true", help="Whether to generate an analysis report")
parser.add_argument("--focus-user", type=str, default=None, help="Focus analysis on a specific user (QQ号)")
args = parser.parse_args()

# 选择设备
device = torch.device("cuda" if args.boost and torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

# 定义 GROUP_ID（用于后续引用）
GROUP_ID = args.group

# 根据模式确定 identifier
if args.mode == "group":
    identifier = args.group
else:
    identifier = args.id

# 1. 数据提取
print("正在提取数据...")
chat_df = extract_chat_data(args.db, identifier, mode=args.mode)
if chat_df.empty:
    print("[ERROR] 未提取到数据，程序退出。")
    exit(1)
print(f"提取到 {len(chat_df)} 条消息记录。")

# 2. 时间范围筛选
print("正在清洗数据...")
print("数据时间范围：", chat_df['timestamp'].min(), "到", chat_df['timestamp'].max())
start_date_str = input("请输入起始日期（例如 2024/01/01）：").strip()
end_date_str = input("请输入结束日期（例如 2024/12/31）：").strip()
if start_date_str:
    try:
        start_date = datetime.strptime(start_date_str, "%Y/%m/%d")
        chat_df = chat_df[chat_df['timestamp'] >= start_date]
        print(f"筛选后，数据起始日期为：{start_date.date()}，剩余 {len(chat_df)} 条记录。")
    except Exception as e:
        print(f"[ERROR] 起始日期解析失败：{e}")
        exit(1)
if end_date_str:
    try:
        end_date = datetime.strptime(end_date_str, "%Y/%m/%d")
        chat_df = chat_df[chat_df['timestamp'] <= end_date]
        print(f"筛选后，数据截止日期为：{end_date.date()}，剩余 {len(chat_df)} 条记录。")
    except Exception as e:
        print(f"[ERROR] 结束日期解析失败：{e}")
        exit(1)
if chat_df.empty:
    print("[ERROR] 筛选后的数据为空，请检查时间范围。")
    exit(1)

# 重置索引
chat_df.reset_index(drop=True, inplace=True)

# 3. 文本嵌入（使用 SentenceTransformer）
print("加载预训练文本嵌入模型...")
text_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
chat_df['text_embedding'] = chat_df['content'].apply(lambda x: text_model.encode(x))

# 4. 计算每个用户的平均文本嵌入（直接均值后归一化）
user_text_embeddings = {}
for user in chat_df['sender_id'].unique():
    embeds = chat_df[chat_df['sender_id'] == user]['text_embedding'].tolist()
    if embeds:
        avg_embed = np.mean(embeds, axis=0)
        norm_val = np.linalg.norm(avg_embed)
        if norm_val > 0:
            user_text_embeddings[user] = avg_embed / norm_val
        else:
            user_text_embeddings[user] = avg_embed
    else:
        user_text_embeddings[user] = np.zeros(text_model.get_sentence_embedding_dimension())

# 5. 计算原始语义相似度矩阵（基于用户平均嵌入）
semantic_matrix = np.zeros((len(user_text_embeddings), len(user_text_embeddings)))
users = list(user_text_embeddings.keys())
num_users = len(users)
for i in range(num_users):
    for j in range(num_users):
        vec_i = user_text_embeddings[users[i]]
        vec_j = user_text_embeddings[users[j]]
        norm_i = np.linalg.norm(vec_i)
        norm_j = np.linalg.norm(vec_j)
        if norm_i > 0 and norm_j > 0:
            semantic_matrix[i, j] = np.dot(vec_i, vec_j) / (norm_i * norm_j)
        else:
            semantic_matrix[i, j] = 0.0

# 5.1 使用基于经验累计分布的离散映射对语义相似度进行赋分
semantic_matrix_mapped = discrete_mapping(semantic_matrix, L=1000)
print("映射后的语义相似度矩阵（前5x5）：")
print(semantic_matrix_mapped[:5, :5])

# 6. 构造行为互动矩阵（统计在5分钟内连续消息的互动次数）
interaction_counts = np.zeros((num_users, num_users), dtype=int)
user_to_index = {user: idx for idx, user in enumerate(users)}
for i in range(len(chat_df) - 1):
    sender_i = chat_df.iloc[i]['sender_id']
    sender_j = chat_df.iloc[i+1]['sender_id']
    t_i = chat_df.iloc[i]['timestamp']
    t_j = chat_df.iloc[i+1]['timestamp']
    if (t_j - t_i).total_seconds() <= 300:
        idx_i = user_to_index[sender_i]
        idx_j = user_to_index[sender_j]
        interaction_counts[idx_i, idx_j] += 1
        interaction_counts[idx_j, idx_i] += 1

# 6.1 对行为矩阵进行离散化映射
behavior_matrix = np.log1p(interaction_counts)
max_behavior = np.max(behavior_matrix)
if max_behavior > 0:
    behavior_norm = behavior_matrix / max_behavior
else:
    behavior_norm = behavior_matrix
# 离散化到 1000 个等级
behavior_norm = np.round(behavior_norm * 999) / 999.0
print("行为得分离散化范围：", behavior_norm.min(), behavior_norm.max())

# 7. 融合行为得分和语义得分（各占50%）
raw_intimacy = 0.5 * behavior_norm + 0.5 * semantic_matrix_mapped
final_intimacy = np.maximum(raw_intimacy, 0)
print("最终亲密度得分范围：", final_intimacy.min(), final_intimacy.max())

# 7.1 定义用户对输出索引（只遍历 i<j），若指定 focus-user 则仅保留包含该用户的对
if args.focus_user:
    focus = str(args.focus_user)
    pair_indices = [(i, j) for i in range(num_users) for j in range(i+1, num_users)
                    if users[i] == focus or users[j] == focus]
else:
    pair_indices = [(i, j) for i in range(num_users) for j in range(i+1, num_users)]

# 保存 CSV 文件（GBK编码），只保存唯一用户对（i<j）
os.makedirs('output', exist_ok=True)
# 构造用户名称映射
user_name_map = {row['sender_id']: row['sender_nickname'] for _, row in chat_df.iterrows()}

csv_path = "output/interaction_scores.csv"
with open(csv_path, 'w', newline='', encoding='gbk') as f:
    writer = csv.writer(f)
    writer.writerow(["UserID1", "UserName1", "UserID2", "UserName2", "BehaviorScore", "SemanticScore", "IntimacyScore"])
    for i, j in pair_indices:
        writer.writerow([
            users[i],
            user_name_map.get(users[i], str(users[i])).encode('gbk', errors='replace').decode('gbk'),
            users[j],
            user_name_map.get(users[j], str(users[j])).encode('gbk', errors='replace').decode('gbk'),
            f"{behavior_norm[i, j]:.4f}",
            f"{semantic_matrix_mapped[i, j]:.4f}",
            f"{final_intimacy[i, j]:.4f}"
        ])

print(f"CSV 文件已保存到 {csv_path}")

# ----- 第8部分：可视化和报告生成 -----
def filter_for_gbk(text: str) -> str:
    return text.encode('gbk', errors='replace').decode('gbk')

labels = [filter_for_gbk(user_name_map.get(u, str(u))) for u in users]

# 生成综合亲密度热力图（仅显示上三角数据）
plot_interaction_heatmap(final_intimacy, labels, save_path="output/interaction_heatmap.png")
# 生成行为得分热力图
plot_custom_heatmap(behavior_norm, labels, title="行为得分热力图", save_path="output/behavior_heatmap.png")
# 生成语义相似度热力图
plot_custom_heatmap(semantic_matrix_mapped, labels, title="语义相似度热力图", save_path="output/semantic_heatmap.png")

# 生成网络图：计算边列表（仅遍历 i<j）
edges = []
for i in range(num_users):
    for j in range(i+1, num_users):
        if interaction_counts[i, j] > 0:
            edges.append((i, j, final_intimacy[i, j]))
# 如果指定了 focus-user，则只保留与该用户相关的边
if args.focus_user:
    focus = str(args.focus_user)
    focus_indices = [idx for idx, u in enumerate(users) if u == focus]
    filtered_edges = [(i, j, w) for (i, j, w) in edges if i in focus_indices or j in focus_indices]
    plot_interaction_network(filtered_edges, labels, save_path="output/interaction_network.png")
else:
    plot_interaction_network(edges, labels, save_path="output/interaction_network.png")

# ----- 第9部分：报告生成（如果指定 --report 参数） -----
if args.report:
    import openai
    def generate_report_via_api(api_key, report_content, save_path="output/analysis_report.md"):
        try:
            client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": report_content}
            ]
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=messages,
                stream=False
            )
            report_text = response.choices[0].message.content
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"[INFO] 分析报告已保存到 {save_path}")
        except Exception as e:
            print(f"[ERROR] 调用 API 生成报告时出错: {e}")

    # 构造 Markdown 格式的用户映射表（放在报告开头）
    user_mapping_str = "## 用户映射表\n\n| 索引 | QQ号 | 昵称 |\n| --- | --- | --- |\n"
    for idx, u in enumerate(users):
        user_mapping_str += f"| {idx} | {u} | {user_name_map.get(u, str(u))} |\n"

    report_content = (
        f"{user_mapping_str}\n"
        f"请根据以下数据生成详细的互动分析报告：\n"
        f"群聊号码：{GROUP_ID}\n"
        f"总用户数：{num_users}\n"
        f"总消息数：{len(chat_df)}\n"
        f"行为矩阵：{np.array(behavior_matrix).tolist()}\n"
        f"文本内容相似度矩阵：{np.array(semantic_matrix).tolist()}\n"
    )
    api_key = "Your_API_KEY"  # 请替换为实际 API Key
    generate_report_via_api(api_key, report_content, save_path="output/analysis_report.md")
else:
    print("[INFO] 未指定 --report 参数，跳过生成分析报告。")

print(f"训练完成。结果已保存至 output/interaction_scores.csv，图表和报告存储在 output/ 目录下。")
