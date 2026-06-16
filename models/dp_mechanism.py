"""
DPLightGCN: 差分隐私LightGCN模型
- 核心创新：层级自适应隐私预算分配
- DP机制：嵌入级高斯扰动
- 隐私会计：Rényi DP Composition
"""

import torch
import torch.nn as nn
import numpy as np
from math import log, sqrt


class EmbeddingPerturbation:
    """嵌入扰动模块：在每层传播后对嵌入加噪"""
    def __init__(self, epsilon, delta, sensitivity, budget_strategy="adaptive"):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.budget_strategy = budget_strategy

    def compute_layer_budgets(self, K, embeddings=None):
        """计算每层的隐私预算分配"""
        if self.budget_strategy == "uniform":
            return [self.epsilon / K] * K
        elif self.budget_strategy == "decreasing":
            weights = [1.0 / (i + 1) for i in range(1, K + 1)]
            total = sum(weights)
            return [self.epsilon * w / total for w in weights]
        elif self.budget_strategy == "adaptive":
            if embeddings is None:
                return [self.epsilon / K] * K
            norms = [torch.norm(e, p=2).item() for e in embeddings]
            norm_sum = sum(norms) + 1e-8
            weights = [n / norm_sum for n in norms]
            return [self.epsilon * w for w in weights]
        else:
            raise ValueError(f"Unknown budget strategy: {self.budget_strategy}")

    def add_noise(self, embedding, eps_layer):
        """对单层嵌入添加高斯噪声
        Args:
            embedding: (n_users+n_items) x emb_dim
            eps_layer: 当前层的隐私预算
        Returns:
            加噪后的嵌入
        """
        sigma = self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / (eps_layer + 1e-8)
        noise = torch.normal(0, sigma, size=embedding.shape, device=embedding.device)
        return embedding + noise

    def get_noise_scale(self, eps_layer):
        """获取当前层的噪声规模（用于绘图和日志）"""
        return self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / (eps_layer + 1e-8)


class RenyiAccountant:
    """基于Rényi DP的隐私会计"""
    def __init__(self, delta=1e-5):
        self.delta = delta
        self.rdp_orders = [alpha for alpha in range(2, 64, 2)]

    def gaussian_rdp(self, sigma, order):
        """高斯机制的RDP计算"""
        return order / (2 * sigma ** 2)

    def compute_total_epsilon(self, sigmas):
        """计算composition后的总ε"""
        best_eps = float("inf")
        for alpha in self.rdp_orders:
            rdp_sum = sum([self.gaussian_rdp(s, alpha) for s in sigmas])
            eps = rdp_sum + log(1.0 / self.delta) / (alpha - 1)
            if eps < best_eps:
                best_eps = eps
        return best_eps


class DPLightGCN(nn.Module):
    """差分隐私LightGCN模型
    Args:
        n_users: 用户数
        n_items: 物品数
        emb_dim: 嵌入维度
        n_layers: GCN层数
        epsilon: 总隐私预算
        delta: DP delta参数
        budget_strategy: 预算分配策略 (uniform/decreasing/adaptive)
    """
    def __init__(self, n_users, n_items, emb_dim, n_layers,
                 epsilon=10.0, delta=1e-5, budget_strategy="adaptive"):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.epsilon = epsilon
        self.delta = delta

        # 嵌入层（用户+物品）
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)

        # 初始化（同LightGCN官方：正态分布）
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        # DP扰动模块
        self.perturbator = EmbeddingPerturbation(
            epsilon=epsilon, delta=delta,
            sensitivity=1.0, budget_strategy=budget_strategy,
        )

        self.epsilon = epsilon  # 存储实际值

    def forward(self, adj_matrix):
        """前向传播（含DP加噪）
        Args:
            adj_matrix: 归一化的邻接矩阵，稀疏张量
        Returns:
            final_embeddings: 加噪后的最终嵌入 (n_users+n_items) x emb_dim
        """
        # 初始嵌入拼接
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embeddings = [ego_embeddings]

        # 第一遍传播：收集各层嵌入（用于自适应预算计算）
        layer_embeddings = []
        temp_emb = ego_embeddings
        for _ in range(self.n_layers):
            temp_emb = torch.sparse.mm(adj_matrix, temp_emb)
            layer_embeddings.append(temp_emb)

        # 计算层级预算
        eps_layers = self.perturbator.compute_layer_budgets(
            self.n_layers, layer_embeddings
        )

        # 第二遍传播：加噪并收集
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj_matrix, ego_embeddings)
            if self.training:
                ego_embeddings = self.perturbator.add_noise(ego_embeddings, eps_layers[k])
            all_embeddings.append(ego_embeddings)

        # 所有层的平均池化（LightGCN标准做法）
        final_embeddings = torch.mean(torch.stack(all_embeddings), dim=0)
        return final_embeddings

    def bpr_loss(self, users, pos_items, neg_items, adj_matrix):
        """贝叶斯个性化排序损失 + L2正则
        Args:
            users: (batch_size,)
            pos_items: (batch_size,)
            neg_items: (batch_size,)
            adj_matrix: 邻接矩阵
        Returns:
            loss: 标量损失
        """
        all_embeddings = self.forward(adj_matrix)
        user_emb = all_embeddings[users]
        pos_emb = all_embeddings[self.n_users + pos_items]
        neg_emb = all_embeddings[self.n_users + neg_items]

        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)

        # BPR损失
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))

        # L2正则
        reg_loss = (1/2) * (
            user_emb.norm(2).pow(2) +
            pos_emb.norm(2).pow(2) +
            neg_emb.norm(2).pow(2)
        ) / users.shape[0]

        return loss + reg_loss

    def get_noise_statistics(self):
        """返回噪声统计信息，用于论文图表"""
        K = self.n_layers
        eps_layers = self.perturbator.compute_layer_budgets(
            K, [torch.randn(10, self.emb_dim) for _ in range(K)]
        )
        sigmas = [self.perturbator.get_noise_scale(eps) for eps in eps_layers]
        accountant = RenyiAccountant(delta=self.delta)
        total_eps = accountant.compute_total_epsilon(sigmas)
        return {
            "layer_epsilons": eps_layers,
            "layer_sigmas": sigmas,
            "total_epsilon_rdp": total_eps,
        }
