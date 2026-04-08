"""
train.py
--------
基于深度学习与文本内容特征融合，分析群聊/私聊中用户互动活跃度。
"""

import os
import csv
import json
import math
import re
import warnings
import numpy as np
from collections import Counter, deque
import pandas as pd
import torch
import networkx as nx
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from extract_chat_data import extract_chat_data, build_display_name_map
from visualization import plot_interaction_network, plot_custom_heatmap, plot_focus_row_heatmap, plot_top_pairs, filter_for_gbk

os.environ.setdefault("HF_HUB_OFFLINE", "1")
warnings.filterwarnings("ignore", message="You are sending unauthenticated requests to the HF Hub.*")


def discrete_mapping(matrix, L=1000):
    triu_indices = np.triu_indices_from(matrix, k=1)
    values = matrix[triu_indices].astype(float)
    if values.size == 0:
        return np.copy(matrix)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return np.nan_to_num(np.copy(matrix), nan=0.0, posinf=0.0, neginf=0.0)
    if not finite_mask.all():
        values = values[finite_mask]
        triu_indices = (triu_indices[0][finite_mask], triu_indices[1][finite_mask])
    jitter = np.random.uniform(-1e-6, 1e-6, size=values.shape)
    values_jitter = values + jitter
    sorted_vals = np.sort(values_jitter)
    if sorted_vals.size == 0:
        return np.nan_to_num(np.copy(matrix), nan=0.0, posinf=0.0, neginf=0.0)
    mapped_continuous = np.interp(values_jitter, sorted_vals, np.linspace(0, 1, len(sorted_vals)))
    discrete_values = np.round(mapped_continuous * (L - 1)) / (L - 1)
    mapped_matrix = np.nan_to_num(np.copy(matrix), nan=0.0, posinf=0.0, neginf=0.0)
    for idx, (i, j) in enumerate(zip(triu_indices[0], triu_indices[1])):
        mapped_matrix[i, j] = discrete_values[idx]
        mapped_matrix[j, i] = discrete_values[idx]
    return mapped_matrix


def optimize_weights(sem, beh, net):
    from scipy.optimize import minimize
    triu_idx = np.triu_indices_from(sem, k=1)
    if triu_idx[0].size == 0:
        return np.array([0.4, 0.4, 0.2])
    def objective(w):
        final = w[0] * sem + w[1] * beh + w[2] * net
        return -np.var(final[triu_idx])
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0.1, 0.6)] * 3
    w0 = np.array([1/3, 1/3, 1/3])
    res = minimize(objective, w0, bounds=bounds, constraints=cons)
    if res.success:
        return res.x
    print("[WARN] 自动调整权重未成功，使用默认权重。")
    return np.array([0.4, 0.4, 0.2])


def balance_score(a, b):
    a = float(a)
    b = float(b)
    if a <= 0 and b <= 0:
        return 0.0
    hi = max(a, b)
    lo = min(a, b)
    return lo / hi if hi > 0 else 0.0


def latency_score(avg_seconds):
    if avg_seconds is None or not np.isfinite(avg_seconds):
        return 0.0
    return float(np.exp(-avg_seconds / 1800.0))


def extract_lexical_tokens(text):
    return re.findall(r'[\u4e00-\u9fff]{1,4}|[A-Za-z0-9_]+|[!?！？~～]{2,}|哈哈+|233+|666+|草+', text.lower())


def weighted_jaccard(counter_a, counter_b):
    keys = set(counter_a) | set(counter_b)
    if not keys:
        return 0.0
    num = sum(min(counter_a.get(k, 0), counter_b.get(k, 0)) for k in keys)
    den = sum(max(counter_a.get(k, 0), counter_b.get(k, 0)) for k in keys)
    return num / den if den else 0.0


def compute_c2c_private_metrics(chat_df, identifier, users, user_to_index):
    self_user = str(identifier)
    num_users = len(users)
    behavior_norm = np.zeros((num_users, num_users))
    semantic_matrix_mapped = np.zeros((num_users, num_users))
    stickiness_matrix = np.zeros((num_users, num_users))
    total_days = max(1, (chat_df['timestamp'].max().date() - chat_df['timestamp'].min().date()).days + 1)

    for peer_id, thread in chat_df.groupby(chat_df['peer_id'].astype(str)):
        peer_id = str(peer_id)
        if peer_id == self_user or peer_id not in user_to_index or self_user not in user_to_index:
            continue
        thread = thread.sort_values('timestamp').reset_index(drop=True)
        self_msgs = thread[thread['sender_id'] == self_user]
        peer_msgs = thread[thread['sender_id'] == peer_id]
        if self_msgs.empty or peer_msgs.empty:
            continue

        msg_sym = balance_score(len(self_msgs), len(peer_msgs))
        len_sym = balance_score(
            self_msgs['content'].astype(str).str.len().mean(),
            peer_msgs['content'].astype(str).str.len().mean()
        )

        reply_deltas = []
        topic_sims = []
        initiations = Counter()
        thread_tokens = {self_user: Counter(), peer_id: Counter()}

        for sender, msgs in ((self_user, self_msgs), (peer_id, peer_msgs)):
            for content in msgs['content'].astype(str):
                thread_tokens[sender].update(extract_lexical_tokens(content))

        for i in range(1, len(thread)):
            prev = thread.iloc[i - 1]
            curr = thread.iloc[i]
            gap = (curr['timestamp'] - prev['timestamp']).total_seconds()
            if gap > 6 * 3600:
                initiations[curr['sender_id']] += 1
            if curr['sender_id'] != prev['sender_id']:
                reply_deltas.append(gap)
                emb_prev = prev['text_embedding']
                emb_curr = curr['text_embedding']
                n_prev = np.linalg.norm(emb_prev)
                n_curr = np.linalg.norm(emb_curr)
                if n_prev > 0 and n_curr > 0:
                    sim = float(np.dot(emb_prev / n_prev, emb_curr / n_curr))
                    topic_sims.append((sim + 1.0) / 2.0)
        if not thread.empty:
            initiations[thread.iloc[0]['sender_id']] += 1

        latency = float(np.mean(reply_deltas)) if reply_deltas else None
        latency_component = latency_score(latency)
        initiation_balance = balance_score(initiations[self_user], initiations[peer_id])
        behavior_score = (
            0.35 * msg_sym +
            0.20 * len_sym +
            0.25 * latency_component +
            0.20 * initiation_balance
        )

        topic_component = float(np.mean(topic_sims)) if topic_sims else 0.0
        lexical_component = weighted_jaccard(thread_tokens[self_user], thread_tokens[peer_id])
        semantic_score = 0.75 * topic_component + 0.25 * lexical_component

        active_days = thread['timestamp'].dt.date.nunique()
        active_days_ratio = active_days / total_days
        night_ratio = float(((thread['timestamp'].dt.hour >= 23) | (thread['timestamp'].dt.hour < 4)).mean())
        night_score = min(1.0, night_ratio / 0.35) if night_ratio > 0 else 0.0
        stickiness_score = 0.7 * active_days_ratio + 0.3 * night_score

        i = user_to_index[self_user]
        j = user_to_index[peer_id]
        behavior_norm[i, j] = behavior_norm[j, i] = behavior_score
        semantic_matrix_mapped[i, j] = semantic_matrix_mapped[j, i] = semantic_score
        stickiness_matrix[i, j] = stickiness_matrix[j, i] = stickiness_score

    behavior_norm = np.round(np.clip(behavior_norm, 0.0, 1.0) * 999) / 999.0
    semantic_matrix_mapped = np.round(np.clip(semantic_matrix_mapped, 0.0, 1.0) * 999) / 999.0
    stickiness_matrix = np.round(np.clip(stickiness_matrix, 0.0, 1.0) * 999) / 999.0
    return behavior_norm, semantic_matrix_mapped, stickiness_matrix


def encode_batch(texts, model, batch_size=32):
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False)


def load_sentence_transformer(model_name, device=None):
    from sentence_transformers import SentenceTransformer
    kwargs = {}
    if device is not None:
        kwargs["device"] = str(device)
    return SentenceTransformer(model_name, local_files_only=True, **kwargs)


def process_chunk(chunk_df, model_name, batch_size):
    model = load_sentence_transformer(model_name)
    texts = chunk_df['content'].tolist()
    embeddings = encode_batch(texts, model, batch_size=batch_size)
    chunk_df = chunk_df.copy()
    chunk_df['text_embedding'] = list(embeddings)
    return chunk_df


def parallel_encode(chat_df, model_name, batch_size=32, num_workers=4, device='cpu'):
    if str(device) != 'cpu':
        model = load_sentence_transformer(model_name, device=device)
        texts = chat_df['content'].tolist()
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        chat_df = chat_df.copy()
        chat_df['text_embedding'] = list(embeddings)
        return chat_df
    n = len(chat_df)
    chunk_size = math.ceil(n / num_workers)
    chunks = [chat_df.iloc[i:i+chunk_size] for i in range(0, n, chunk_size)]
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(lambda c: process_chunk(c, model_name, batch_size), chunks))
    return pd.concat(results).reset_index(drop=True)


def generate_report_via_api(api_key, report_content, save_path="output/analysis_report.md",
                            system_prompt="You are a helpful assistant."):
    import openai
    try:
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
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
            raw = ask("我的 QQ 号（将分析与所有私聊好友的互动）")
            if raw and raw.isdigit():
                id_ = int(raw)
                break
            print("  请输入有效的 QQ 号。")

    # 聚焦用户（仅群聊模式有意义）
    print()
    focus_user = None
    lite = False
    if mode == "group":
        focus_raw = ask("聚焦分析某用户的 QQ 号（回车跳过）", "")
        focus_user = focus_raw if focus_raw else None
        if focus_user:
            lite = ask_yn("  启用精简模式（仅保留该用户相关互动）")

    # 排除用户
    exclude_raw = ask("排除用户 QQ 号（多个用逗号分隔，回车跳过）", "")
    exclude_users = [qq.strip() for qq in exclude_raw.split(",") if qq.strip().isdigit()] if exclude_raw else []

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
        exclude_users=exclude_users,
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

if args.boost:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用设备：cuda ({torch.cuda.get_device_name(0)})")
    else:
        print(f"[WARN] 启用 GPU 加速失败，回退 CPU。")
        print(f"  torch.__version__     = {torch.__version__}")
        print(f"  torch.version.cuda    = {torch.version.cuda}")
        print(f"  cuda.is_available()   = False")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")
    print(f"使用设备：cpu")

focus_user = str(args.focus_user) if args.focus_user else None
identifier = args.group if args.mode == "group" else args.id
if args.mode == "c2c":
    focus_user = str(identifier)

_now = datetime.now()
mode_dir = "group" if args.mode == "group" else "private"
out_dir = os.path.join("output", _now.strftime("%Y"), _now.strftime("%m"), _now.strftime("%d"), mode_dir, str(identifier))

# 1. 数据提取
print("正在提取数据...")
chat_df = extract_chat_data(args.db, identifier, mode=args.mode, remote=args.remote, cipher_config=cipher_config)
if chat_df.empty:
    print("[ERROR] 未提取到数据，程序退出。")
    exit(1)
print(f"提取到 {len(chat_df)} 条消息记录。")

preferred_display_names = {}
for sender_id, group in chat_df.groupby('sender_id'):
    candidates = []
    for raw_name in group['sender_nickname'].astype(str):
        name = raw_name.strip()
        if not name or name.lower() == 'nan' or name.isdigit():
            continue
        candidates.append(name)
    if candidates:
        ranked = sorted(
            Counter(candidates).items(),
            key=lambda item: (item[1], len(item[0])),
            reverse=True
        )
        preferred_display_names[str(sender_id)] = ranked[0][0]
preferred_display_names[str(identifier)] = "我" if args.mode == "c2c" else preferred_display_names.get(str(identifier), str(identifier))
numeric_only_ids = [sender_id for sender_id, name in preferred_display_names.items() if str(name).isdigit()]
missing_name_ids = [sender_id for sender_id in chat_df['sender_id'].astype(str).unique() if sender_id not in preferred_display_names or preferred_display_names.get(sender_id, "").isdigit()]
alias_overrides = build_display_name_map(args.db, missing_name_ids, remote=args.remote, cipher_config=cipher_config)
preferred_display_names.update(alias_overrides)
preferred_display_names[str(identifier)] = "我" if args.mode == "c2c" else preferred_display_names.get(str(identifier), str(identifier))

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

if args.exclude_users:
    chat_df = chat_df[~chat_df['sender_id'].isin(args.exclude_users)]
    print(f"排除 {args.exclude_users} 后，剩余 {len(chat_df)} 条记录。")
    if chat_df.empty:
        print("[ERROR] 排除后数据为空，请检查 QQ 号是否正确。")
        exit(1)

chat_df = chat_df.sort_values('timestamp').reset_index(drop=True)
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
chat_df = parallel_encode(chat_df, model_name, batch_size=32, num_workers=4, device=device)
print("文本嵌入计算完成。")

# 4. 用户列表与索引
users = list(chat_df['sender_id'].unique())
num_users = len(users)
user_to_index = {user: idx for idx, user in enumerate(users)}

if num_users < 2:
    print(f"[WARN] 当前筛选后的数据仅包含 {num_users} 个用户，无法形成用户对互动矩阵。")
    print("[WARN] 这通常说明私聊提取逻辑没有正确区分自己和对方，或当前时间范围内只剩单方消息。")

if args.mode == "c2c":
    print("计算私聊行为特征得分...")
    behavior_norm, semantic_matrix_mapped, third_matrix = compute_c2c_private_metrics(
        chat_df, identifier, users, user_to_index
    )
    third_metric_name = "时间粘性"
    behavior_title = "行为特征得分热力图"
    semantic_title = "语义上下文得分热力图"
    third_title = "时间粘性得分热力图"
    print("行为特征得分范围：", behavior_norm.min(), behavior_norm.max())
    print("语义上下文得分（前5x5）：")
    print(semantic_matrix_mapped[:5, :5])
    print("时间粘性得分范围：", third_matrix.min(), third_matrix.max())
    interaction_weights = behavior_norm.copy()
else:
    # 5. 语义相似度矩阵（基于 5 分钟窗口内实际交互消息对的余弦相似度均值）
    print("计算语义相似度矩阵（交互对）...")
    sem_scores = np.zeros((num_users, num_users))
    sem_counts = np.zeros((num_users, num_users))
    win = deque()  # (timestamp, sender_idx, normed_embedding, peer_id)
    for _, row in chat_df.iterrows():
        t_i = row['timestamp']
        s_i = user_to_index[row['sender_id']]
        peer_i = None
        emb_i = row['text_embedding']
        n_i = np.linalg.norm(emb_i)
        emb_i_n = emb_i / n_i if n_i > 0 else emb_i
        while win and (t_i - win[0][0]).total_seconds() > 300:
            win.popleft()
        for t_j, s_j, emb_j_n, peer_j in win:
            if s_j != s_i:
                sim = float(np.dot(emb_i_n, emb_j_n))
                sem_scores[s_i, s_j] += sim
                sem_scores[s_j, s_i] += sim
                sem_counts[s_i, s_j] += 1
                sem_counts[s_j, s_i] += 1
        win.append((t_i, s_i, emb_i_n, peer_i))
    semantic_matrix = np.divide(sem_scores, sem_counts, out=np.zeros_like(sem_scores), where=sem_counts > 0)
    semantic_matrix_mapped = discrete_mapping(semantic_matrix, L=1000)
    print("语义相似度矩阵（前5x5）：")
    print(semantic_matrix_mapped[:5, :5])

    # 6. 行为互动矩阵（5 分钟滑动窗口 + 指数时间衰减）
    print("计算行为互动矩阵（滑动窗口）...")
    TAU = 150.0  # 衰减时间常数（秒）
    interaction_weights = np.zeros((num_users, num_users))
    win = deque()  # (timestamp, sender_idx, peer_id)
    for _, row in chat_df.iterrows():
        t_i = row['timestamp']
        s_i = user_to_index[row['sender_id']]
        while win and (t_i - win[0][0]).total_seconds() > 300:
            win.popleft()
        for t_j, s_j, peer_j in win:
            if s_j != s_i:
                w = math.exp(-(t_i - t_j).total_seconds() / TAU)
                interaction_weights[s_i, s_j] += w
                interaction_weights[s_j, s_i] += w
        win.append((t_i, s_i, None))
    behavior_matrix = np.log1p(interaction_weights)
    max_behavior = np.max(behavior_matrix)
    behavior_norm = behavior_matrix / max_behavior if max_behavior > 0 else behavior_matrix
    behavior_norm = np.round(behavior_norm * 999) / 999.0
    print("行为得分离散化范围：", behavior_norm.min(), behavior_norm.max())

    # 7. 群聊网络拓扑
    third_metric_name = "网络拓扑"
    behavior_title = "行为得分热力图"
    semantic_title = "语义相似度热力图"
    third_title = "网络拓扑得分热力图"
    G = nx.Graph()
    G.add_nodes_from(range(num_users))
    for i in range(num_users):
        for j in range(i+1, num_users):
            if interaction_weights[i, j] > 0:
                G.add_edge(i, j, weight=float(interaction_weights[i, j]))
    third_matrix = np.zeros((num_users, num_users))
    for i in range(num_users):
        for j in range(i+1, num_users):
            nbrs_i = set(G.neighbors(i))
            nbrs_j = set(G.neighbors(j))
            union = nbrs_i | nbrs_j
            jac = len(nbrs_i & nbrs_j) / len(union) if union else 0.0
            third_matrix[i, j] = jac
            third_matrix[j, i] = jac

# 8. 用户名称映射与标签
os.makedirs(out_dir, exist_ok=True)
user_name_map = {}
for _, row in chat_df.iterrows():
    nickname = row['sender_nickname']
    if pd.isna(nickname) or str(nickname).strip() == "":
        nickname = str(row['sender_id'])
    sender_id = str(row['sender_id'])
    user_name_map[sender_id] = preferred_display_names.get(sender_id, str(nickname))
labels = [filter_for_gbk(user_name_map.get(u, str(u))) for u in users]

mapping_path = f"{out_dir}/user_mapping.txt"
with open(mapping_path, 'w', encoding='gbk', errors='replace') as f:
    f.write("索引\tQQ号\t昵称\n")
    for idx, u in enumerate(users):
        f.write(f"{idx}\t{u}\t{user_name_map.get(u, str(u))}\n")
print(f"用户映射文件已保存到 {mapping_path}")

# 9. 可视化
if num_users >= 2:
    if args.mode == "c2c" and str(identifier) in user_to_index:
        focus_idx = user_to_index[str(identifier)]
        plot_focus_row_heatmap(semantic_matrix_mapped, labels, focus_idx, title=semantic_title + time_range_str, save_path=f"{out_dir}/semantic_heatmap{time_range_str}.png")
        plot_focus_row_heatmap(behavior_norm, labels, focus_idx, title=behavior_title + time_range_str, save_path=f"{out_dir}/behavior_heatmap{time_range_str}.png")
        plot_focus_row_heatmap(third_matrix, labels, focus_idx, title=third_title + time_range_str, save_path=f"{out_dir}/network_heatmap{time_range_str}.png")
    else:
        plot_custom_heatmap(semantic_matrix_mapped, labels, title=semantic_title + time_range_str, save_path=f"{out_dir}/semantic_heatmap{time_range_str}.png")
        plot_custom_heatmap(behavior_norm, labels, title=behavior_title + time_range_str, save_path=f"{out_dir}/behavior_heatmap{time_range_str}.png")
        plot_custom_heatmap(third_matrix, labels, title=third_title + time_range_str, save_path=f"{out_dir}/network_heatmap{time_range_str}.png")
else:
    print("[WARN] 用户数不足 2，跳过语义/行为/网络热力图生成。")

edges = [(i, j, (third_matrix[i,j] + semantic_matrix_mapped[i,j] + behavior_norm[i,j]) / 3)
         for i in range(num_users) for j in range(i+1, num_users) if interaction_weights[i, j] > 0]
if focus_user:
    focus_indices = {idx for idx, u in enumerate(users) if u == focus_user}
    edges = [(i, j, w) for i, j, w in edges if i in focus_indices or j in focus_indices]
if edges:
    plot_interaction_network(edges, labels, save_path=f"{out_dir}/interaction_network{time_range_str}.png")
else:
    print("[WARN] 没有可绘制的互动边，跳过网络图生成。")

# 10. 融合指标
if args.mode == "c2c":
    w_sem, w_beh, w_net = 0.4, 0.4, 0.2
    print(f"私聊模式使用固定权重：语义上下文 {w_sem:.2f}, 行为特征 {w_beh:.2f}, 时间粘性 {w_net:.2f}")
elif args.auto_weight:
    w_sem, w_beh, w_net = optimize_weights(semantic_matrix_mapped, behavior_norm, third_matrix)
    print(f"自动调整权重：语义 {w_sem:.2f}, 行为 {w_beh:.2f}, {third_metric_name} {w_net:.2f}")
else:
    w_sem, w_beh, w_net = 0.4, 0.4, 0.2
final_intimacy = np.maximum(w_sem * semantic_matrix_mapped + w_beh * behavior_norm + w_net * third_matrix, 0)
print("最终互动活跃度得分范围：", final_intimacy.min(), final_intimacy.max())

if num_users >= 2:
    if args.mode == "c2c" and str(identifier) in user_to_index:
        plot_focus_row_heatmap(final_intimacy, labels, user_to_index[str(identifier)], title="综合亲密度热力图" + time_range_str,
                               save_path=f"{out_dir}/intimacy_heatmap{time_range_str}.png")
    else:
        plot_custom_heatmap(final_intimacy, labels, title="综合亲密度热力图" + time_range_str,
                            save_path=f"{out_dir}/intimacy_heatmap{time_range_str}.png")
    plot_top_pairs(final_intimacy, behavior_norm, semantic_matrix_mapped, third_matrix,
                   users, user_name_map,
                   save_path=f"{out_dir}/top_pairs{time_range_str}.png",
                   third_label=third_metric_name)
else:
    print("[WARN] 用户数不足 2，跳过综合亲密度热力图和 Top 互动对图。")

# 11. CSV 输出
if focus_user:
    pair_indices = [(i, j) for i in range(num_users) for j in range(i+1, num_users)
                    if users[i] == focus_user or users[j] == focus_user]
else:
    pair_indices = [(i, j) for i in range(num_users) for j in range(i+1, num_users)]
csv_path = f"{out_dir}/interaction_scores{time_range_str}.csv"
with open(csv_path, 'w', newline='', encoding='gbk') as f:
    writer = csv.writer(f)
    third_score_header = "NetworkScore" if args.mode == "group" else "StickinessScore"
    behavior_score_header = "BehaviorScore" if args.mode == "group" else "BehavioralScore"
    semantic_score_header = "SemanticScore" if args.mode == "group" else "SemanticContextScore"
    writer.writerow(["UserID1", "UserName1", "UserID2", "UserName2", behavior_score_header, semantic_score_header, third_score_header, "IntimacyScore"])
    for i, j in pair_indices:
        writer.writerow([
            users[i], filter_for_gbk(user_name_map.get(users[i], str(users[i]))),
            users[j], filter_for_gbk(user_name_map.get(users[j], str(users[j]))),
            f"{behavior_norm[i,j]:.4f}", f"{semantic_matrix_mapped[i,j]:.4f}",
            f"{third_matrix[i,j]:.4f}", f"{final_intimacy[i,j]:.4f}"
        ])
print(f"CSV 文件已保存到 {csv_path}")

# 12. 报告生成
if args.report:
    # 用户映射 + 消息量
    msg_counts = chat_df['sender_id'].value_counts()
    user_table = "## 用户列表\n\n| 昵称 | QQ号 | 消息数 |\n| --- | --- | --- |\n"
    for u in users:
        user_table += f"| {user_name_map.get(u, u)} | {u} | {msg_counts.get(u, 0)} |\n"

    # Top 20 互动对
    pair_scores = []
    for i in range(num_users):
        for j in range(i + 1, num_users):
            if final_intimacy[i, j] > 0:
                pair_scores.append((
                    user_name_map.get(users[i], users[i]),
                    user_name_map.get(users[j], users[j]),
                    final_intimacy[i, j],
                    behavior_norm[i, j],
                    semantic_matrix_mapped[i, j],
                    third_matrix[i, j],
                ))
    pair_scores.sort(key=lambda x: x[2], reverse=True)
    third_table_label = "网络" if args.mode == "group" else "消息量"
    top_table = f"## Top 20 互动对\n\n| 用户A | 用户B | 亲密度 | 行为 | 语义 | {third_table_label} |\n| --- | --- | --- | --- | --- | --- |\n"
    for na, nb, intim, beh, sem, third_score in pair_scores[:20]:
        top_table += f"| {na} | {nb} | {intim:.3f} | {beh:.3f} | {sem:.3f} | {third_score:.3f} |\n"

    if args.mode == "group":
        scope_desc = (f"聚焦用户 {user_name_map.get(focus_user, focus_user)} 与其他成员的互动"
                      if focus_user else "群内所有成员互动")
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
        scope_desc = (f"聚焦用户 {user_name_map.get(focus_user, focus_user)} 与其他私聊对象的互动"
                      if focus_user else f"账号 {args.id} 与所有私聊对象的互动")
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

    user_prompt = f"""请根据以下 {analysis_title}，撰写一份结构化的人际关系分析报告。

## 分析背景
{analysis_object}
- 数据范围：{chat_df['timestamp'].min().strftime('%Y/%m/%d')} 至 {chat_df['timestamp'].max().strftime('%Y/%m/%d')}
- 总用户数：{num_users}，总消息数：{len(chat_df)}
- 分析范围：{scope_desc}
- 融合权重：语义 {w_sem:.2f} / 行为 {w_beh:.2f} / {third_metric_name} {w_net:.2f}

## 指标说明
- **行为得分**：综合互动对称性、响应延迟与主动发起平衡度，反映互动是否双向、及时、均衡
- **语义得分**：综合相邻问答语义连贯性与高频词/语气词共现，反映双方是否处于同一语境
{metric_desc}
- **亲密度**：三项加权融合（0-1），综合衡量互动关系强度

{user_table}

{top_table}

{report_requirements}
"""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    generate_report_via_api(api_key, user_prompt, save_path=f"{out_dir}/analysis_report.md", system_prompt=system_prompt)
else:
    print("[INFO] 未选择生成分析报告，跳过。")

print(f"分析完成。结果已保存至 {out_dir}/")
