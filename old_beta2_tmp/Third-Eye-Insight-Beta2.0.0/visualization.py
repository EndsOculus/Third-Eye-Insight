"""visualization.py
生成用户互动关系可视化图表的模块。
"""

import matplotlib.pyplot as plt
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
    fig, ax = plt.subplots(figsize=(max(8, len(labels)*0.5), max(6, len(labels)*0.5)))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("得分", rotation=-90, va="bottom")
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.2f}"))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] {title} 已保存到 {save_path}")


def plot_interaction_network(edge_list, labels, save_path="interaction_network.png"):
    labels = process_labels(labels)
    G = nx.Graph()
    num_users = len(labels)
    for idx, name in enumerate(labels):
        G.add_node(idx, name=name)
    for (i, j, score) in edge_list:
        G.add_edge(i, j, weight=score)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(max(8, num_users * 0.5), max(6, num_users * 0.5)))
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=800)
    nx.draw_networkx_labels(G, pos, labels={i: labels[i] for i in range(num_users)}, font_size=10)
    if edge_list:
        weights = [score for (_, _, score) in edge_list]
        min_w, max_w = min(weights), max(weights)
        widths = [2] * len(weights) if max_w == min_w else [1 + 4 * ((w - min_w) / (max_w - min_w)) for w in weights]
        nx.draw_networkx_edges(G, pos, width=widths, edge_color='gray')
    else:
        nx.draw_networkx_edges(G, pos)
    plt.title("用户互动网络图")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"网络图已保存到 {save_path}")
