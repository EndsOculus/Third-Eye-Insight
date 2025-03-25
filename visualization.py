"""visualization.py
生成用户互动关系可视化图表的模块。
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import re

# 全局字体设置（微软雅黑及备用字体）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def filter_for_gbk(text: str) -> str:
    """
    将文本转换为 GBK 可显示格式，不可显示的字符用 '?' 替换。
    """
    return text.encode('gbk', errors='replace').decode('gbk')

def truncate_label(text: str, max_len: int = 10) -> str:
    """
    如果文本长度超过 max_len，则截断后追加省略号。
    """
    return text if len(text) <= max_len else text[:max_len] + "..."

def process_labels(labels: list) -> list:
    """
    对标签列表先将其转换为 GBK 可显示格式，再截断超过 10 个字符的部分。
    """
    processed = [filter_for_gbk(label) for label in labels]
    processed = [truncate_label(label, 10) for label in processed]
    return processed

def plot_interaction_heatmap(matrix, labels, save_path="interaction_heatmap.png"):
    """
    绘制融合后最终亲密度热力图（仅显示唯一用户对，上三角矩阵）。
    """
    labels = process_labels(labels)
    # 对矩阵只保留上三角部分
    triu_matrix = np.triu(matrix, k=1)
    width = max(8, len(labels) * 0.5)
    height = max(6, len(labels) * 0.5)
    plt.figure(figsize=(width, height))
    sns.heatmap(triu_matrix, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, cmap="YlOrRd", mask=(triu_matrix==0))
    plt.title("用户互动亲密度热力图")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"热力图已保存到 {save_path}")

def plot_custom_heatmap(matrix, labels, title, save_path="heatmap.png"):
    """
    绘制自定义热力图，用于显示行为得分或语义相似度。
    仅显示上三角矩阵中的数据（i<j）。
    """
    labels = process_labels(labels)  # 先对标签进行过滤和截断处理
    # 取上三角（不包含对角线，k=1）
    triu_matrix = np.triu(matrix, k=1)
    # 仅对非零部分进行色彩映射（避免重复用户对显示，用斜杠合并可在 CSV 中体现）
    plt.figure(figsize=(max(8, len(labels) * 0.5), max(6, len(labels) * 0.5)))
    sns.heatmap(triu_matrix, annot=True, fmt=".6f", cmap="YlOrRd", mask=(triu_matrix==0), vmin=0, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"{title} 已保存到 {save_path}")

def plot_interaction_network(edge_list, labels, save_path="interaction_network.png"):
    """
    绘制用户互动网络图。
    参数：
        edge_list (list of tuples): 每个元组为 (i, j, score) 表示用户 i 和 j 之间的亲密度得分。
        labels (list): 用户标签列表。
        save_path (str): 网络图保存路径。
    """
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
        if max_w == min_w:
            widths = [2] * len(weights)
        else:
            widths = [1 + 4 * ((w - min_w) / (max_w - min_w)) for w in weights]
        nx.draw_networkx_edges(G, pos, width=widths, edge_color='gray')
    else:
        nx.draw_networkx_edges(G, pos)
    plt.title("用户互动网络图")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"网络图已保存到 {save_path}")
