"""
train.py
--------
基于深度学习与文本内容特征融合，分析群聊/私聊中用户互动活跃度。
"""

import os
import csv
import json
import math
import numpy as np
import pandas as pd
import torch
import networkx as nx
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from scipy.optimize import minimize
from sentence_transformers import SentenceTransformer
from extract_chat_data import extract_chat_data
from visualization import plot_interaction_network, plot_custom_heatmap, filter_for_gbk


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


def optimize_weights(sem, beh, net):
    def objective(w):
        final = w[0] * sem + w[1] * beh + w[2] * net
        triu_idx = np.triu_indices_from(final, k=1)
        return -np.var(final[triu_idx])
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0.2, 0.4)] * 3
    w0 = np.array([0.45, 0.45, 0.1])
    res = minimize(objective, w0, bounds=bounds, constraints=cons)
    if res.success:
        return res.x
    print("[WARN] 自动调整权重未成功，使用默认权重。")
    return w0


def encode_batch(texts, model, batch_size=32):
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False)


def process_chunk(chunk_df, model_name, batch_size):
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
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(lambda c: process_chunk(c, model_name, batch_size), chunks))
    return pd.concat(results).reset_index(drop=True)


def generate_report_via_api(api_key, report_content, save_path="output/analysis_report.md"):
    import openai
    try:
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": report_content}
            ],
            stream=False
        )
        report_text = response.choices[0].message.content
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"[INFO] 分析报告已保存到 {save_path}")
    except Exception as e:
        print(f"[ERROR] 调用 API 生成报告时出错: {e}")


def ask(prompt, default=None):
    """带默认值的输入，直接回车返回 default。"""
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else default


def ask_yn(prompt, default=False):
    """y/N 选择，返回 bool。"""
    hint = "[Y/n]" if default else "[y/N]"
    val = input(f"{prompt} {hint} ").strip().lower()
    if not val:
        return default
    return val == 'y'


def interactive_config(cipher_config: dict) -> SimpleNamespace:
    """交互式收集运行参数，返回配置对象。"""
    print("\n======== 觉之瞳 · 群聊互动分析 ========\n")

    # 数据库路径
    default_db = cipher_config.get('db_file', 'nt_msg.clean.db')
    db = ask("数据库文件路径", default_db)

    # 模式
    print("\n分析模式：")
    print("  1. 群聊")
    print("  2. 私聊")
    mode_choice = ask("请选择", "1")
    mode = "c2c" if mode_choice.strip() == "2" else "group"

    # 群号 / QQ 号
    if mode == "group":
        while True:
            raw = ask("群号")
            if raw and raw.isdigit():
                group = int(raw)
                break
            print("  请输入有效的群号。")
        id_ = group
    else:
        group = 0
        while True:
            raw = ask("好友 QQ 号")
            if raw and raw.isdigit():
                id_ = int(raw)
                break
            print("  请输入有效的 QQ 号。")

    # 聚焦用户
    print()
    focus_raw = ask("聚焦分析某用户的 QQ 号（回车跳过）", "")
    focus_user = focus_raw if focus_raw else None
    lite = False
    if focus_user:
        lite = ask_yn("  启用精简模式（仅保留该用户相关互动）")

    # 可选功能
    print()
    remote     = ask_yn("使用远程数据库连接")
    boost      = ask_yn("启用 GPU 加速")
    auto_wt    = ask_yn("自动调整融合权重")
    report     = ask_yn("生成 AI 分析报告（需 DEEPSEEK_API_KEY 环境变量）")
    font       = ask("图表中文字体", "Microsoft YaHei")

    print()
    return SimpleNamespace(
        db=db, mode=mode, group=group, id=id_,
        focus_user=focus_user, lite=lite,
        remote=remote, boost=boost,
        auto_weight=auto_wt, report=report, font=font,
    )


# ── 启动 ──────────────────────────────────────────────────────────────────────

# 加载加密配置
cipher_config = {}
if os.path.exists('config.json'):
    with open('config.json', 'r', encoding='utf-8') as _f:
        cipher_config = json.load(_f)

# 检测 nt_msg.db，提示剥离文件头
if os.path.exists('nt_msg.db'):
    if ask_yn("发现 nt_msg.db，是否剥离 1024 字节文件头生成 nt_msg.clean_e.db"):
        with open('nt_msg.db', 'rb') as _fin:
            _fin.seek(1024)
            _stripped = _fin.read()
        with open('nt_msg.clean_e.db', 'wb') as _fout:
            _fout.write(_stripped)
        del _stripped
        print("[OK] nt_msg.clean_e.db 已生成。")
        if ask_yn("是否删除 nt_msg.db"):
            os.remove('nt_msg.db')
            print("[OK] nt_msg.db 已删除。")

# 交互式参数收集
args = interactive_config(cipher_config)

device = torch.device("cuda" if args.boost and torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

focus_user = str(args.focus_user) if args.focus_user else None
identifier = args.group if args.mode == "group" else args.id

# 1. 数据提取
print("正在提取数据...")
chat_df = extract_chat_data(args.db, identifier, mode=args.mode, remote=args.remote, cipher_config=cipher_config)
if chat_df.empty:
    print("[ERROR] 未提取到数据，程序退出。")
    exit(1)
print(f"提取到 {len(chat_df)} 条消息记录。")

# 2. 时间范围筛选
print("正在清洗数据...")
print("数据时间范围：", chat_df['timestamp'].min(), "到", chat_df['timestamp'].max())
start_date_str = input("请输入起始日期（例如 2024/01/01），直接回车表示不限制：").strip()
end_date_str   = input("请输入结束日期（例如 2024/12/31），直接回车表示不限制：").strip()
if start_date_str:
    try:
        start_date = datetime.strptime(start_date_str, "%Y/%m/%d")
        chat_df = chat_df[chat_df['timestamp'] >= start_date]
        print(f"筛选后起始日期：{start_date.date()}，剩余 {len(chat_df)} 条记录。")
    except Exception as e:
        print(f"[ERROR] 起始日期解析失败：{e}")
        exit(1)
if end_date_str:
    try:
        end_date = datetime.strptime(end_date_str, "%Y/%m/%d")
        chat_df = chat_df[chat_df['timestamp'] <= end_date]
        print(f"筛选后截止日期：{end_date.date()}，剩余 {len(chat_df)} 条记录。")
    except Exception as e:
        print(f"[ERROR] 结束日期解析失败：{e}")
        exit(1)
if chat_df.empty:
    print("[ERROR] 筛选后数据为空，请检查时间范围。")
    exit(1)
chat_df.reset_index(drop=True, inplace=True)
time_range_str = ""
if start_date_str or end_date_str:
    s = start_date_str.replace("/", "-") if start_date_str else "start"
    e = end_date_str.replace("/", "-") if end_date_str else "end"
    time_range_str = f"_{s}-{e}"

# lite 模式：仅保留与 focus-user 相关的互动记录
if args.lite and focus_user:
    related_indices = set()
    for i in range(len(chat_df) - 1):
        s_cur = chat_df.iloc[i]['sender_id']
        s_nxt = chat_df.iloc[i+1]['sender_id']
        t_cur = chat_df.iloc[i]['timestamp']
        t_nxt = chat_df.iloc[i+1]['timestamp']
        if (s_cur == focus_user or s_nxt == focus_user) and (t_nxt - t_cur).total_seconds() <= 300:
            related_indices.add(i)
            related_indices.add(i+1)
    chat_df = chat_df.iloc[sorted(related_indices)].reset_index(drop=True)
    print(f"仅保留与用户 {focus_user} 相关的互动记录，共 {len(chat_df)} 条记录。")

# 3. 文本嵌入
print("加载预训练文本嵌入模型并进行批量计算...")
model_name = "paraphrase-multilingual-MiniLM-L12-v2"
chat_df = parallel_encode(chat_df, model_name, batch_size=32, num_workers=4)
print("文本嵌入计算完成。")

# 4. 计算每个用户的平均文本嵌入（均值后归一化）
user_text_embeddings = {}
for user in chat_df['sender_id'].unique():
    embeds = chat_df[chat_df['sender_id'] == user]['text_embedding'].tolist()
    avg_embed = np.mean(embeds, axis=0)
    norm_val = np.linalg.norm(avg_embed)
    user_text_embeddings[user] = avg_embed / norm_val if norm_val > 0 else avg_embed

# 5. 计算语义相似度矩阵（向量化余弦相似度，嵌入已归一化）
users = list(user_text_embeddings.keys())
num_users = len(users)
emb_matrix = np.array([user_text_embeddings[u] for u in users])
semantic_matrix = emb_matrix @ emb_matrix.T
semantic_matrix_mapped = discrete_mapping(semantic_matrix, L=1000)
print("映射后的语义相似度矩阵（前5x5）：")
print(semantic_matrix_mapped[:5, :5])

# 6. 构造行为互动矩阵（5分钟内连续消息互动次数）
interaction_counts = np.zeros((num_users, num_users), dtype=int)
user_to_index = {user: idx for idx, user in enumerate(users)}
for i in range(len(chat_df) - 1):
    s_i = chat_df.iloc[i]['sender_id']
    s_j = chat_df.iloc[i+1]['sender_id']
    if s_i == s_j:
        continue
    t_i = chat_df.iloc[i]['timestamp']
    t_j = chat_df.iloc[i+1]['timestamp']
    if (t_j - t_i).total_seconds() <= 300:
        idx_i = user_to_index[s_i]
        idx_j = user_to_index[s_j]
        interaction_counts[idx_i, idx_j] += 1
        interaction_counts[idx_j, idx_i] += 1
behavior_matrix = np.log1p(interaction_counts)
max_behavior = np.max(behavior_matrix)
behavior_norm = behavior_matrix / max_behavior if max_behavior > 0 else behavior_matrix
behavior_norm = np.round(behavior_norm * 999) / 999.0
print("行为得分离散化范围：", behavior_norm.min(), behavior_norm.max())

# 7. 网络拓扑分析：度中心性
G = nx.Graph()
G.add_nodes_from(users)
for i in range(num_users):
    for j in range(i+1, num_users):
        if interaction_counts[i, j] > 0:
            G.add_edge(users[i], users[j], weight=int(interaction_counts[i, j]))
net_centrality = nx.degree_centrality(G)
centrality_values = np.array([net_centrality.get(u, 0) for u in users])
network_matrix = (centrality_values[:, None] + centrality_values[None, :]) / 2

# 8. 用户名称映射与标签
os.makedirs('output', exist_ok=True)
user_name_map = {row['sender_id']: row['sender_nickname'] for _, row in chat_df.iterrows()}
labels = [filter_for_gbk(user_name_map.get(u, str(u))) for u in users]

mapping_path = "output/user_mapping.txt"
with open(mapping_path, 'w', encoding='gbk', errors='replace') as f:
    f.write("索引\tQQ号\t昵称\n")
    for idx, u in enumerate(users):
        f.write(f"{idx}\t{u}\t{user_name_map.get(u, str(u))}\n")
print(f"用户映射文件已保存到 {mapping_path}")

# 9. 可视化
plot_custom_heatmap(semantic_matrix_mapped, labels, title="语义相似度热力图" + time_range_str, save_path=f"output/semantic_heatmap{time_range_str}.png")
plot_custom_heatmap(behavior_norm, labels, title="行为得分热力图" + time_range_str, save_path=f"output/behavior_heatmap{time_range_str}.png")
plot_custom_heatmap(network_matrix, labels, title="网络拓扑得分热力图" + time_range_str, save_path=f"output/network_heatmap{time_range_str}.png")

edges = [(i, j, (network_matrix[i,j] + semantic_matrix_mapped[i,j] + behavior_norm[i,j]) / 3)
         for i in range(num_users) for j in range(i+1, num_users) if interaction_counts[i, j] > 0]
if focus_user:
    focus_indices = {idx for idx, u in enumerate(users) if u == focus_user}
    edges = [(i, j, w) for i, j, w in edges if i in focus_indices or j in focus_indices]
plot_interaction_network(edges, labels, save_path=f"output/interaction_network{time_range_str}.png")

# 10. 融合指标
if args.auto_weight:
    w_sem, w_beh, w_net = optimize_weights(semantic_matrix_mapped, behavior_norm, network_matrix)
    print(f"自动调整权重：语义 {w_sem:.2f}, 行为 {w_beh:.2f}, 网络 {w_net:.2f}")
else:
    w_sem, w_beh, w_net = 0.4, 0.4, 0.2
final_intimacy = np.maximum(w_sem * semantic_matrix_mapped + w_beh * behavior_norm + w_net * network_matrix, 0)
print("最终互动活跃度得分范围：", final_intimacy.min(), final_intimacy.max())

# 11. CSV 输出
if focus_user:
    pair_indices = [(i, j) for i in range(num_users) for j in range(i+1, num_users)
                    if users[i] == focus_user or users[j] == focus_user]
else:
    pair_indices = [(i, j) for i in range(num_users) for j in range(i+1, num_users)]
csv_path = f"output/interaction_scores{time_range_str}.csv"
with open(csv_path, 'w', newline='', encoding='gbk') as f:
    writer = csv.writer(f)
    writer.writerow(["UserID1", "UserName1", "UserID2", "UserName2", "BehaviorScore", "SemanticScore", "NetworkScore", "IntimacyScore"])
    for i, j in pair_indices:
        writer.writerow([
            users[i], filter_for_gbk(user_name_map.get(users[i], str(users[i]))),
            users[j], filter_for_gbk(user_name_map.get(users[j], str(users[j]))),
            f"{behavior_norm[i,j]:.4f}", f"{semantic_matrix_mapped[i,j]:.4f}",
            f"{network_matrix[i,j]:.4f}", f"{final_intimacy[i,j]:.4f}"
        ])
print(f"CSV 文件已保存到 {csv_path}")

# 12. 报告生成
if args.report:
    user_mapping_str = "## 用户映射表\n\n| 索引 | QQ号 | 昵称 |\n| --- | --- | --- |\n"
    for idx, u in enumerate(users):
        user_mapping_str += f"| {idx} | {u} | {user_name_map.get(u, str(u))} |\n"
    report_prefix = f"以下数据仅包含群聊中 QQ 号为 {focus_user} 的用户与其他用户之间的互动记录。" if focus_user else "以下数据基于群聊中所有用户的聊天内容。"
    report_content = (
        f"{user_mapping_str}\n{report_prefix}\n"
        f"群聊号码：{args.group}\n"
        f"数据时间范围：{chat_df['timestamp'].min().strftime('%Y/%m/%d')} 至 {chat_df['timestamp'].max().strftime('%Y/%m/%d')}\n"
        f"总用户数：{num_users}\n总消息数：{len(chat_df)}\n"
        f"行为矩阵：{behavior_matrix.tolist()}\n"
        f"文本内容相似度矩阵：{semantic_matrix.tolist()}\n"
    )
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    generate_report_via_api(api_key, report_content)
else:
    print("[INFO] 未选择生成分析报告，跳过。")

print(f"分析完成。结果已保存至 output/ 目录下。")
