"""model.py
定义用于用户互动关系表示的深度学习模型。
"""
import torch
import torch.nn as nn

class InteractionModel(nn.Module):
    def __init__(self, num_users: int, embedding_dim: int = 16):
        """
        初始化模型，为每个用户创建一个嵌入向量。
        参数：
            num_users (int): 用户数量。
            embedding_dim (int): 嵌入向量的维度。
        """
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
    
    def forward(self, user_index_i, user_index_j):
        """
        前向传播：计算两用户嵌入向量的点积作为互动得分。
        参数：
            user_index_i, user_index_j: 用户索引（张量）
        返回：
            用户嵌入向量的点积得分（张量）
        """
        emb_i = self.user_embedding(user_index_i)
        emb_j = self.user_embedding(user_index_j)
        score = (emb_i * emb_j).sum(dim=-1)
        return score
