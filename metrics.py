import logging
import math
import re
from collections import Counter, deque

import networkx as nx
import numpy as np
from tqdm import tqdm

from analysis_types import MetricArtifacts

logger = logging.getLogger(__name__)


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

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = [(0.1, 0.6)] * 3
    w0 = np.array([1 / 3, 1 / 3, 1 / 3])
    res = minimize(objective, w0, bounds=bounds, constraints=cons)
    if res.success:
        return res.x
    logger.warning("自动调整权重未成功，使用默认权重。")
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
    return re.findall(r"[\u4e00-\u9fff]{1,4}|[A-Za-z0-9_]+|[!?！？~～]{2,}|哈哈+|233+|666+|草+", text.lower())


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
    total_days = max(1, (chat_df["timestamp"].max().date() - chat_df["timestamp"].min().date()).days + 1)

    for peer_id, thread in tqdm(chat_df.groupby(chat_df["peer_id"].astype(str)), desc="私聊指标", unit="对", leave=True):
        peer_id = str(peer_id)
        if peer_id == self_user or peer_id not in user_to_index or self_user not in user_to_index:
            continue
        thread = thread.sort_values("timestamp").reset_index(drop=True)
        self_msgs = thread[thread["sender_id"] == self_user]
        peer_msgs = thread[thread["sender_id"] == peer_id]
        if self_msgs.empty or peer_msgs.empty:
            continue

        msg_sym = balance_score(len(self_msgs), len(peer_msgs))
        len_sym = balance_score(self_msgs["content"].str.len().mean(), peer_msgs["content"].str.len().mean())

        reply_deltas = []
        topic_sims = []
        initiations = Counter()
        thread_tokens = {self_user: Counter(), peer_id: Counter()}

        for sender, msgs in ((self_user, self_msgs), (peer_id, peer_msgs)):
            for content in msgs["content"]:
                thread_tokens[sender].update(extract_lexical_tokens(content))

        for i in range(1, len(thread)):
            prev = thread.iloc[i - 1]
            curr = thread.iloc[i]
            gap = (curr["timestamp"] - prev["timestamp"]).total_seconds()
            if gap > 6 * 3600:
                initiations[curr["sender_id"]] += 1
            if curr["sender_id"] != prev["sender_id"]:
                reply_deltas.append(gap)
                emb_prev = prev["text_embedding"]
                emb_curr = curr["text_embedding"]
                n_prev = np.linalg.norm(emb_prev)
                n_curr = np.linalg.norm(emb_curr)
                if n_prev > 0 and n_curr > 0:
                    sim = float(np.dot(emb_prev / n_prev, emb_curr / n_curr))
                    topic_sims.append((sim + 1.0) / 2.0)
        if not thread.empty:
            initiations[thread.iloc[0]["sender_id"]] += 1

        latency = float(np.mean(reply_deltas)) if reply_deltas else None
        behavior_score = (
            0.35 * msg_sym
            + 0.20 * len_sym
            + 0.25 * latency_score(latency)
            + 0.20 * balance_score(initiations[self_user], initiations[peer_id])
        )
        semantic_score = 0.75 * (float(np.mean(topic_sims)) if topic_sims else 0.0) + 0.25 * weighted_jaccard(
            thread_tokens[self_user], thread_tokens[peer_id]
        )

        active_days_ratio = thread["timestamp"].dt.date.nunique() / total_days
        night_ratio = float(((thread["timestamp"].dt.hour >= 23) | (thread["timestamp"].dt.hour < 4)).mean())
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


def compute_group_semantic_matrix(chat_df, user_to_index, window_size, L):
    sem_scores = np.zeros((len(user_to_index), len(user_to_index)))
    sem_counts = np.zeros((len(user_to_index), len(user_to_index)))
    win = deque()
    for row in tqdm(chat_df.itertuples(index=False), desc="语义矩阵", unit="条", total=len(chat_df), leave=True):
        timestamp = row.timestamp
        sender_idx = user_to_index[str(row.sender_id)]
        embedding = row.text_embedding
        norm = np.linalg.norm(embedding)
        normalized_embedding = embedding / norm if norm > 0 else embedding
        while win and (timestamp - win[0][0]).total_seconds() > window_size:
            win.popleft()
        for _, prev_sender_idx, prev_embedding in win:
            if prev_sender_idx != sender_idx:
                sim = float(np.dot(normalized_embedding, prev_embedding))
                sem_scores[sender_idx, prev_sender_idx] += sim
                sem_scores[prev_sender_idx, sender_idx] += sim
                sem_counts[sender_idx, prev_sender_idx] += 1
                sem_counts[prev_sender_idx, sender_idx] += 1
        win.append((timestamp, sender_idx, normalized_embedding))
    semantic_matrix = np.divide(sem_scores, sem_counts, out=np.zeros_like(sem_scores), where=sem_counts > 0)
    return discrete_mapping(semantic_matrix, L=L)


def compute_group_behavior_matrix(chat_df, user_to_index, window_size, tau):
    interaction_weights = np.zeros((len(user_to_index), len(user_to_index)))
    win = deque()
    for row in tqdm(chat_df.itertuples(index=False), desc="行为矩阵", unit="条", total=len(chat_df), leave=True):
        timestamp = row.timestamp
        sender_idx = user_to_index[str(row.sender_id)]
        while win and (timestamp - win[0][0]).total_seconds() > window_size:
            win.popleft()
        for prev_timestamp, prev_sender_idx in win:
            if prev_sender_idx != sender_idx:
                weight = math.exp(-(timestamp - prev_timestamp).total_seconds() / tau)
                interaction_weights[sender_idx, prev_sender_idx] += weight
                interaction_weights[prev_sender_idx, sender_idx] += weight
        win.append((timestamp, sender_idx))
    behavior_matrix = np.log1p(interaction_weights)
    max_behavior = np.max(behavior_matrix)
    behavior_norm = behavior_matrix / max_behavior if max_behavior > 0 else behavior_matrix
    return np.round(behavior_norm * 999) / 999.0, interaction_weights


def compute_group_network_matrix(interaction_weights):
    num_users = interaction_weights.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(num_users))
    for i in range(num_users):
        for j in range(i + 1, num_users):
            if interaction_weights[i, j] > 0:
                graph.add_edge(i, j, weight=float(interaction_weights[i, j]))

    network_matrix = np.zeros((num_users, num_users))
    pairs = [(i, j) for i in range(num_users) for j in range(i + 1, num_users)]
    for i, j in tqdm(pairs, desc="网络矩阵", unit="对", leave=True):
        neighbors_i = set(graph.neighbors(i))
        neighbors_j = set(graph.neighbors(j))
        union = neighbors_i | neighbors_j
        jaccard = len(neighbors_i & neighbors_j) / len(union) if union else 0.0
        network_matrix[i, j] = network_matrix[j, i] = jaccard
    return network_matrix


def compute_metrics(chat_df, args, identifier, users, user_to_index, analysis_params):
    if args.mode == "c2c":
        logger.info("计算私聊行为特征得分...")
        behavior_norm, semantic_matrix_mapped, third_matrix = compute_c2c_private_metrics(chat_df, identifier, users, user_to_index)
        return MetricArtifacts(
            behavior_norm=behavior_norm,
            semantic_matrix_mapped=semantic_matrix_mapped,
            third_matrix=third_matrix,
            interaction_weights=behavior_norm.copy(),
            third_metric_name="时间粘性",
            behavior_title="行为特征得分热力图",
            semantic_title="语义上下文得分热力图",
            third_title="时间粘性得分热力图",
        )

    logger.info("计算语义相似度矩阵（交互对）...")
    semantic_matrix_mapped = compute_group_semantic_matrix(chat_df, user_to_index, analysis_params.WINDOW_SIZE, analysis_params.L)
    logger.debug(f"语义相似度矩阵（前5x5）：\n{semantic_matrix_mapped[:5, :5]}")

    logger.info("计算行为互动矩阵（滑动窗口）...")
    behavior_norm, interaction_weights = compute_group_behavior_matrix(chat_df, user_to_index, analysis_params.WINDOW_SIZE, analysis_params.TAU)
    logger.debug(f"行为得分离散化范围：{behavior_norm.min():.4f} - {behavior_norm.max():.4f}")

    logger.info("计算网络拓扑矩阵...")
    third_matrix = compute_group_network_matrix(interaction_weights)

    return MetricArtifacts(
        behavior_norm=behavior_norm,
        semantic_matrix_mapped=semantic_matrix_mapped,
        third_matrix=third_matrix,
        interaction_weights=interaction_weights,
        third_metric_name="网络拓扑",
        behavior_title="行为得分热力图",
        semantic_title="语义相似度热力图",
        third_title="网络拓扑得分热力图",
    )


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
