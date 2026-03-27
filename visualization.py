"""visualization.py
生成用户互动关系可视化图表的模块。
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import networkx as nx
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def filter_for_gbk(text: str) -> str:
    return text.encode('gbk', errors='replace').decode('gbk')


def truncate_label(text: str, max_len: int = 10) -> str:
    return text if len(text) <= max_len else text[:max_len] + "..."


def process_labels(labels: list) -> list:
    return [truncate_label(filter_for_gbk(label), 10) for label in labels]


def plot_custom_heatmap(matrix, labels, title, save_path):
    """
    热力图：按行总分降序排列行列（活跃用户聚在左上角），对角线遮蔽。
    人数 <= 30 时显示数值注释。
    """
    n = len(labels)

    # 按行总分（不含对角线）重排，让活跃用户聚在左上角
    row_sums = matrix.sum(axis=1) - np.diag(matrix)
    order = np.argsort(row_sums)[::-1]
    matrix_sorted = matrix[np.ix_(order, order)]
    labels_sorted = [labels[i] for i in order]

    cell = max(0.5, min(1.0, 20 / n))
    fig, ax = plt.subplots(figsize=(max(8, n * cell + 2), max(6, n * cell + 1)))

    mask = np.eye(n, dtype=bool)
    tick_fs   = max(6, min(11, 120 // n))
    annot_fs  = max(5, min(8,  90 // n))

    sns.heatmap(
        matrix_sorted,
        mask=mask,
        ax=ax,
        cmap='viridis',
        xticklabels=labels_sorted,
        yticklabels=labels_sorted,
        annot=(n <= 30),
        fmt='.2f',
        annot_kws={'size': annot_fs},
        cbar_kws={'label': 'score', 'shrink': 0.8},
        linewidths=0.2 if n <= 40 else 0,
        linecolor='#555555',
        square=True,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=tick_fs)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tick_fs)
    ax.set_title(title + "（按活跃度排序）", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] {title} 已保存到 {save_path}")


def plot_top_pairs(final_intimacy, behavior_norm, semantic_matrix_mapped,
                   network_matrix, users, user_name_map, save_path, top_n=20):
    """
    水平堆叠条形图：展示互动得分最高的 top_n 对用户，
    拆分显示行为/语义/网络三项分量，一眼可见谁和谁最亲密、得分来源。
    """
    n = len(users)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            score = final_intimacy[i, j]
            if score > 0:
                name_i = filter_for_gbk(truncate_label(user_name_map.get(users[i], str(users[i])), 10))
                name_j = filter_for_gbk(truncate_label(user_name_map.get(users[j], str(users[j])), 10))
                pairs.append((
                    f"{name_i} & {name_j}",
                    behavior_norm[i, j],
                    semantic_matrix_mapped[i, j],
                    network_matrix[i, j],
                    score,
                ))
    if not pairs:
        return
    pairs.sort(key=lambda x: x[4], reverse=True)
    pairs = pairs[:top_n]
    pairs.reverse()  # 条形图从下往上画，反转让最高分在顶部

    labels  = [p[0] for p in pairs]
    beh     = np.array([p[1] for p in pairs])
    sem     = np.array([p[2] for p in pairs])
    net     = np.array([p[3] for p in pairs])

    fig_h = max(5, len(pairs) * 0.42 + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    y = np.arange(len(pairs))
    bar_h = 0.55
    ax.barh(y, beh, bar_h, label='行为互动', color='#4c72b0')
    ax.barh(y, sem, bar_h, left=beh, label='语义相似', color='#55a868')
    ax.barh(y, net, bar_h, left=beh + sem, label='网络拓扑', color='#c44e52')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('得分', fontsize=10)
    ax.set_title(f'Top {len(pairs)} 互动对（亲密度分量拆解）', fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.set_xlim(0, max((beh + sem + net).max() * 1.05, 0.1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Top 互动对条形图已保存到 {save_path}")


def plot_interaction_network(edge_list, labels, save_path="interaction_network.png"):
    """
    网络图：节点大小/颜色编码互动权重之和，边宽度/透明度编码亲密度强弱。
    """
    labels = process_labels(labels)
    G = nx.Graph()
    num_users = len(labels)
    for idx in range(num_users):
        G.add_node(idx, name=labels[idx])
    for (i, j, score) in edge_list:
        G.add_edge(i, j, weight=score)

    pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(num_users + 1))

    weighted_deg = {i: sum(d['weight'] for _, _, d in G.edges(i, data=True))
                    for i in range(num_users)}
    max_wd = max(weighted_deg.values()) if any(v > 0 for v in weighted_deg.values()) else 1.0
    node_sizes  = [400 + 1400 * (weighted_deg.get(i, 0) / max_wd) for i in range(num_users)]
    node_colors = [weighted_deg.get(i, 0) for i in range(num_users)]

    fig_sz = max(8, num_users * 0.7)
    fig, ax = plt.subplots(figsize=(fig_sz, fig_sz * 0.85))

    nc = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                                node_size=node_sizes, cmap='viridis', alpha=0.9)
    plt.colorbar(nc, ax=ax, label='互动权重之和', shrink=0.6, pad=0.02)

    label_fs = max(6, min(10, 100 // num_users))
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={i: labels[i] for i in range(num_users)},
                            font_size=label_fs)

    if edge_list:
        weights = [score for (_, _, score) in edge_list]
        min_w, max_w = min(weights), max(weights)
        rng = (max_w - min_w) or 1.0
        for (i, j, score) in edge_list:
            nw = (score - min_w) / rng
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(i, j)],
                                   width=0.5 + 3.5 * nw,
                                   edge_color='steelblue',
                                   alpha=0.15 + 0.75 * nw)
    else:
        nx.draw_networkx_edges(G, pos, ax=ax)

    ax.set_title("用户互动网络图", fontsize=13)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 网络图已保存到 {save_path}")
