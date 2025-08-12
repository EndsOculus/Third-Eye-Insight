"""
train.py
--------
基于深度学习与文本内容特征融合，分析群聊/私聊中用户互动活跃度。
功能包括：
1. 命令行参数支持：--group, --db, --mode, --id, --font, --boost, --report, --focus-user, --lite, --auto-weight, --remote
2. 交互式输入起始和结束日期进行时间筛选（格式：YYYY/MM/DD）。
3. 使用 SentenceTransformer 提取文本嵌入，并计算每个用户的平均嵌入（均值后归一化）。
4. 构造行为矩阵（统计5分钟内连续消息的互动次数），对 np.log1p(interaction_counts) 归一化后离散化映射。
5. 计算语义相似度矩阵（基于余弦相似度），并使用 ECDF 离散化映射函数拉大差异。
6. 基于 NetworkX 构造网络拓扑指标（度中心性），定义网络得分为二者平均。
7. 自动调整权重（语义、行为、网络）或使用默认权重，融合三项指标得到最终互动活跃度得分。
8. 输出 CSV 文件（GBK编码）、各指标热力图和网络图，文件名中包含时间区间；另外生成用户映射文件。
9. 若指定 --report 参数，则调用 API 生成详细分析报告（报告中包含用户映射表）。
10. 若使用 --lite 参数且指定 --focus-user，则仅保留 focus-user 与其他用户的互动记录。
11. 若指定 --remote 参数，则使用远程数据库连接；否则默认使用本地数据库文件。
"""

import argparse
import numpy as np
import pandas as pd
import torch
from extract_chat_data import extract_chat_data
from sentence_transformers import SentenceTransformer
from datetime import datetime
import csv
import os
import math
import networkx as nx
from visualization import plot_interaction_heatmap, plot_interaction_network, plot_custom_heatmap
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor  # 改为使用 ThreadPoolExecutor

# 定义离散化映射函数（ECDF 离散映射，拉大差异）
def discrete_mapping(matrix, L=1000):
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

# 自动调整权重（仅对语义、行为、网络）
def optimize_weights(sem, beh, net):
    def objective(w):
        final = w[0] * sem + w[1] * beh + w[2] * net
        triu_idx = np.triu_indices_from(final, k=1)
        var_val = np.var(final[triu_idx])
        return -var_val
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0.2, 0.4)] * 3
    w0 = np.array([0.45, 0.45, 0.1])
    res = minimize(objective, w0, bounds=bounds, constraints=cons)
    if res.success:
        return res.x
    else:
        print("[WARN] 自动调整权重未成功，使用默认权重。")
        return w0

# 解析命令行参数
parser = argparse.ArgumentParser(description="Deep Learning Chat Interaction Affinity Analysis")
parser.add_argument("--group", type=int, required=True, help="群聊号码（群聊模式有效）")
parser.add_argument("--db", type=str, required=True, help="数据库连接字符串或本地数据库文件路径")
parser.add_argument("--mode", type=str, choices=["group", "c2c"], required=True, help="模式：group 表示群聊，c2c 表示私聊")
parser.add_argument("--id", type=int, required=False, help="群聊模式下与 --group 相同；私聊模式下为好友 QQ 号")
parser.add_argument("--font", type=str, default="Microsoft YaHei", help="图表使用的中文字体（默认：Microsoft YaHei）")
parser.add_argument("--boost", action="store_true", help="启用 GPU 加速（默认关闭）")
parser.add_argument("--report", action="store_true", help="是否生成分析报告")
parser.add_argument("--focus-user", type=str, default=None, help="仅分析指定用户与其他人的互动（QQ号）")
parser.add_argument("--lite", action="store_true", help="如果设置，则仅基于 focus-user 与其他用户的互动数据")
parser.add_argument("--auto-weight", action="store_true", help="自动调整指标融合权重")
parser.add_argument("--remote", action="store_true", help="使用远程数据库连接（否则默认使用本地数据库）")
args = parser.parse_args()

# 如果是群聊模式，则自动将 --id 设置为 --group
if args.mode == "group":
    args.id = args.group

# 选择设备
device = torch.device("cuda" if args.boost and torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

GROUP_ID = args.group
identifier = args.group if args.mode == "group" else args.id

# 1. 数据提取
print("正在提取数据...")
chat_df = extract_chat_data(args.db, identifier, args.mode, args.remote)
if chat_df.empty:
    print("[ERROR] 未提取到数据，程序退出。")
    exit(1)
print(f"提取到 {len(chat_df)} 条消息记录。")

# 2. 时间范围筛选
print("正在清洗数据...")
print("数据时间范围：", chat_df['timestamp'].min(), "到", chat_df['timestamp'].max())
start_date_str = input("请输入起始日期（例如 2024/01/01），直接回车表示不限制：").strip()
end_date_str = input("请输入结束日期（例如 2024/12/31），直接回车表示不限制：").strip()
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
chat_df.reset_index(drop=True, inplace=True)
time_range_str = ""
# 生成时间区间字符串，并将 "/" 替换为 "-"
if start_date_str or end_date_str:
    # 替换日期中的 "/" 为 "-"
    s = start_date_str.replace("/", "-") if start_date_str else "start"
    e = end_date_str.replace("/", "-") if end_date_str else "end"
    time_range_str = f"_{s}-{e}"

# 若使用 lite 模式且指定了 focus-user，则仅保留与 focus-user 相关的互动记录
if args.lite and args.focus_user:
    focus = str(args.focus_user)
    related_indices = set()
    for i in range(len(chat_df) - 1):
        sender_current = chat_df.iloc[i]['sender_id']
        sender_next = chat_df.iloc[i+1]['sender_id']
        t_current = chat_df.iloc[i]['timestamp']
        t_next = chat_df.iloc[i+1]['timestamp']
        if (sender_current == focus or sender_next == focus) and ((t_next - t_current).total_seconds() <= 300):
            related_indices.add(i)
            related_indices.add(i+1)
    chat_df = chat_df.iloc[list(related_indices)].reset_index(drop=True)
    print(f"仅保留与用户 {focus} 相关的互动记录，共 {len(chat_df)} 条记录。")

# 3. 文本嵌入（使用批量处理+多线程）
print("加载预训练文本嵌入模型并进行批量计算...")
import math
from concurrent.futures import ThreadPoolExecutor

def encode_batch(texts, model, batch_size=32):
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False)

def process_chunk(chunk_df, model_name, batch_size):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    texts = chunk_df['content'].tolist()
    embeddings = encode_batch(texts, model, batch_size=batch_size)
    chunk_df = chunk_df.copy()
    chunk_df['text_embedding'] = list(embeddings)
    return chunk_df

def parallel_encode(chat_df, model_name, batch_size=32, num_workers=4):
    n = len(chat_df)
    chunk_size = math.ceil(n / num_workers)
    chunks = [chat_df.iloc[i:i+chunk_size] for i in range(0, n, chunk_size)]
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_chunk, chunk, model_name, batch_size) for chunk in chunks]
        for future in futures:
            results.append(future.result())
    return pd.concat(results).reset_index(drop=True)

model_name = "paraphrase-multilingual-MiniLM-L12-v2"
batch_size = 32
num_workers = 4
chat_df = parallel_encode(chat_df, model_name, batch_size=batch_size, num_workers=num_workers)
print("文本嵌入计算完成。")

# 4. 计算每个用户的平均文本嵌入（均值后归一化）
user_text_embeddings = {}
for user in chat_df['sender_id'].unique():
    embeds = chat_df[chat_df['sender_id'] == user]['text_embedding'].tolist()
    if embeds:
        avg_embed = np.mean(embeds, axis=0)
        norm_val = np.linalg.norm(avg_embed)
        user_text_embeddings[user] = avg_embed / norm_val if norm_val > 0 else avg_embed
    else:
        user_text_embeddings[user] = np.zeros(text_model.get_sentence_embedding_dimension())

# 5. 计算语义相似度矩阵（基于余弦相似度）并离散化映射
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
semantic_matrix_mapped = discrete_mapping(semantic_matrix, L=1000)
print("映射后的语义相似度矩阵（前5x5）：")
print(semantic_matrix_mapped[:5, :5])

# 6. 构造行为互动矩阵（统计5分钟内连续消息的互动次数）
interaction_counts = np.zeros((num_users, num_users), dtype=int)
user_to_index = {user: idx for idx, user in enumerate(users)}
for i in range(len(chat_df) - 1):
    sender_i = chat_df.iloc[i]['sender_id']
    sender_j = chat_df.iloc[i+1]['sender_id']
    if sender_i == sender_j:
        continue
    t_i = chat_df.iloc[i]['timestamp']
    t_j = chat_df.iloc[i+1]['timestamp']
    if (t_j - t_i).total_seconds() <= 300:
        idx_i = user_to_index[sender_i]
        idx_j = user_to_index[sender_j]
        interaction_counts[idx_i, idx_j] += 1
        interaction_counts[idx_j, idx_i] += 1
behavior_matrix = np.log1p(interaction_counts)
max_behavior = np.max(behavior_matrix)
if max_behavior > 0:
    behavior_norm = behavior_matrix / max_behavior
else:
    behavior_norm = behavior_matrix
behavior_norm = np.round(behavior_norm * 999) / 999.0
print("行为得分离散化范围：", behavior_norm.min(), behavior_norm.max())

# 7. 网络拓扑分析：基于行为互动构建图，并计算度中心性
G = nx.Graph()
for u in users:
    G.add_node(u)
for i in range(num_users):
    for j in range(i+1, num_users):
        if interaction_counts[i, j] > 0:
            G.add_edge(users[i], users[j], weight=interaction_counts[i, j])
net_centrality = nx.degree_centrality(G)
network_matrix = np.zeros((num_users, num_users))
for i in range(num_users):
    for j in range(num_users):
        network_matrix[i, j] = (net_centrality.get(users[i], 0) + net_centrality.get(users[j], 0)) / 2

# 8. 构造用户名称映射（确保 extract_chat_data 返回 sender_nickname 列）
user_name_map = {row['sender_id']: row['sender_nickname'] for _, row in chat_df.iterrows()}
def filter_for_gbk(text: str) -> str:
    return text.encode('gbk', errors='replace').decode('gbk')
labels = [filter_for_gbk(user_name_map.get(u, str(u))) for u in users]

# 输出用户映射文件
mapping_path = "output/user_mapping.txt"
os.makedirs('output', exist_ok=True)
with open(mapping_path, 'w', encoding='gbk', errors='replace') as f:
    f.write("索引\tQQ号\t昵称\n")
    for idx, u in enumerate(users):
        f.write(f"{idx}\t{u}\t{user_name_map.get(u, str(u))}\n")
print(f"用户映射文件已保存到 {mapping_path}")

# 9. 可视化：生成各指标热力图和网络图（文件名包含时间区间）
plot_custom_heatmap(semantic_matrix_mapped, labels, title="语义相似度热力图" + time_range_str, save_path=f"output/semantic_heatmap{time_range_str}.png")
plot_custom_heatmap(behavior_norm, labels, title="行为得分热力图" + time_range_str, save_path=f"output/behavior_heatmap{time_range_str}.png")
plot_custom_heatmap(network_matrix, labels, title="网络拓扑得分热力图" + time_range_str, save_path=f"output/network_heatmap{time_range_str}.png")

edges = []
for i in range(num_users):
    for j in range(i+1, num_users):
        if interaction_counts[i, j] > 0:
            combined_score = (network_matrix[i, j] + semantic_matrix_mapped[i, j] + behavior_norm[i, j]) / 3
            edges.append((i, j, combined_score))
if args.focus_user:
    focus = str(args.focus_user)
    focus_indices = [idx for idx, u in enumerate(users) if u == focus]
    filtered_edges = [(i, j, w) for (i, j, w) in edges if i in focus_indices or j in focus_indices]
    plot_interaction_network(filtered_edges, labels, save_path=f"output/interaction_network{time_range_str}.png")
else:
    plot_interaction_network(edges, labels, save_path=f"output/interaction_network{time_range_str}.png")

# 10. 融合指标：自动调整权重或使用默认权重，得到最终互动活跃度得分
if args.auto_weight:
    weights = optimize_weights(semantic_matrix_mapped, behavior_norm, network_matrix)
    w_sem, w_beh, w_net = weights
    print(f"自动调整权重结果：语义 {w_sem:.2f}, 行为 {w_beh:.2f}, 网络 {w_net:.2f}")
else:
    w_sem, w_beh, w_net = 0.4, 0.4, 0.2
raw_intimacy = w_sem * semantic_matrix_mapped + w_beh * behavior_norm + w_net * network_matrix
final_intimacy = np.maximum(raw_intimacy, 0)
print("最终互动活跃度得分范围：", final_intimacy.min(), final_intimacy.max())

# 11. 生成 CSV 输出（仅保存唯一用户对：i < j）
if args.focus_user:
    focus = str(args.focus_user)
    pair_indices = [(i, j) for i in range(num_users) for j in range(i+1, num_users)
                    if users[i] == focus or users[j] == focus]
else:
    pair_indices = [(i, j) for i in range(num_users) for j in range(i+1, num_users)]
csv_path = f"output/interaction_scores{time_range_str}.csv"
with open(csv_path, 'w', newline='', encoding='gbk') as f:
    writer = csv.writer(f)
    writer.writerow(["UserID1", "UserName1", "UserID2", "UserName2", "BehaviorScore", "SemanticScore", "NetworkScore", "IntimacyScore"])
    for i, j in pair_indices:
        beh = behavior_norm[i, j]
        sem_score = semantic_matrix_mapped[i, j]
        net_score = network_matrix[i, j]
        final_score = final_intimacy[i, j]
        writer.writerow([
            users[i],
            user_name_map.get(users[i], str(users[i])).encode('gbk', errors='replace').decode('gbk'),
            users[j],
            user_name_map.get(users[j], str(users[j])).encode('gbk', errors='replace').decode('gbk'),
            f"{beh:.4f}",
            f"{sem_score:.4f}",
            f"{net_score:.4f}",
            f"{final_score:.4f}"
        ])
print(f"CSV 文件已保存到 {csv_path}")

# 12. 报告生成（若指定 --report 参数）
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
    user_mapping_str = "## 用户映射表\n\n| 索引 | QQ号 | 昵称 |\n| --- | --- | --- |\n"
    for idx, u in enumerate(users):
        user_mapping_str += f"| {idx} | {u} | {user_name_map.get(u, str(u))} |\n"
    report_prefix = f"以下数据仅包含群聊中 QQ 号为 {args.focus_user} 的用户与其他用户之间的互动记录。" if args.focus_user else "以下数据基于群聊中所有用户的聊天内容。"
    report_content = (
        f"{user_mapping_str}\n"
        f"{report_prefix}\n"
        f"群聊号码：{GROUP_ID}\n"
        f"数据时间范围：{chat_df['timestamp'].min().strftime('%Y/%m/%d')} 至 {chat_df['timestamp'].max().strftime('%Y/%m/%d')}\n"
        f"总用户数：{num_users}\n"
        f"总消息数：{len(chat_df)}\n"
        f"行为矩阵：{np.array(behavior_matrix).tolist()}\n"
        f"文本内容相似度矩阵：{np.array(semantic_matrix).tolist()}\n"
    )
    api_key = "YOUR_API_KEY"  # 请替换为实际 API Key
    generate_report_via_api(api_key, report_content, save_path="output/analysis_report.md")
else:
    print("[INFO] 未指定 --report 参数，跳过生成分析报告。")

print(f"训练完成。结果已保存至 {csv_path}，图表和报告存储在 output/ 目录下。")
