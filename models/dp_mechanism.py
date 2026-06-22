"""
DPLightGCN: 差分隐私LightGCN模型
- 自适应裁剪 + 嵌入级高斯扰动 + RDP隐私核算
- 三种预算分配策略：uniform / decreasing / adaptive
"""
import torch
import torch.nn as nn
import numpy as np
from math import log


class EmbeddingPerturbation:
    """嵌入扰动模块：裁剪 + 加噪 + 预算分配"""
    def __init__(self, epsilon, delta, budget_strategy="adaptive"):
        self.epsilon = epsilon
        self.delta = delta
        self.budget_strategy = budget_strategy

    def compute_layer_budgets(self, K, embeddings=None):
        """将总预算 epsilon 分配到 K 层"""
        if self.budget_strategy == "uniform":
            return [self.epsilon / K] * K
        elif self.budget_strategy == "decreasing":
            weights = [1.0 / (i + 1) for i in range(1, K + 1)]
            total = sum(weights)
            return [self.epsilon * w / total for w in weights]
        elif self.budget_strategy == "adaptive":
            if embeddings is None or any(e is None for e in embeddings):
                return [self.epsilon / K] * K
            # 使用 Frobenius 范数衡量每层信息密度
            norms = [torch.norm(e.detach(), p="fro").item() for e in embeddings]
            norm_sum = max(sum(norms), 1e-8)
            weights = [n / norm_sum for n in norms]
            return [self.epsilon * w for w in weights]
        else:
            raise ValueError(f"Unknown budget strategy: {self.budget_strategy}")

    def get_clip_threshold(self, embedding):
        """自适应裁剪阈值：平均范数 x 2，最低 0.1"""
        avg_norm = torch.norm(embedding.detach(), p=2, dim=1).mean().item()
        return max(avg_norm * 2.0, 0.1)

    def clip_and_noise(self, embedding, eps_layer):
        """裁剪嵌入 + 添加高斯噪声"""
        clip_C = self.get_clip_threshold(embedding)
        norms = torch.norm(embedding, p=2, dim=1, keepdim=True)
        scaling = torch.clamp(clip_C / (norms + 1e-8), max=1.0)
        clipped = embedding * scaling
        sigma = clip_C * np.sqrt(2 * np.log(1.25 / self.delta)) / (eps_layer + 1e-8)
        noise = torch.normal(0, sigma, size=clipped.shape, device=clipped.device)
        return clipped + noise


class DPLightGCN(nn.Module):
    """差分隐私 LightGCN 模型"""
    def __init__(self, n_users, n_items, emb_dim, n_layers,
                 epsilon=10.0, delta=1e-5, budget_strategy="adaptive"):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.epsilon = epsilon
        self.delta = delta
        self.budget_strategy = budget_strategy

        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        self.perturbator = EmbeddingPerturbation(
            epsilon=epsilon, delta=delta, budget_strategy=budget_strategy)

    def forward(self, adj_matrix, apply_dp=True):
        """前向传播：图传播 + 可选 DP 扰动"""
        ego = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_emb = [ego]
        layer_emb = []
        tmp = ego

        with torch.no_grad():
            for _ in range(self.n_layers):
                tmp = torch.sparse.mm(adj_matrix, tmp)
                layer_emb.append(tmp)

        eps_layers = self.perturbator.compute_layer_budgets(self.n_layers, layer_emb)

        for k in range(self.n_layers):
            ego = torch.sparse.mm(adj_matrix, ego)
            if apply_dp and self.training:
                ego = self.perturbator.clip_and_noise(ego, eps_layers[k])
            all_emb.append(ego)

        return torch.mean(torch.stack(all_emb), dim=0)

    def bpr_loss(self, users, pos_items, neg_items, adj_matrix, apply_dp=True):
        """BPR 损失 + L2 正则化"""
        all_emb = self.forward(adj_matrix, apply_dp=apply_dp)
        u_e = all_emb[users]
        p_e = all_emb[self.n_users + pos_items]
        n_e = all_emb[self.n_users + neg_items]
        pos_scores = torch.sum(u_e * p_e, dim=1)
        neg_scores = torch.sum(u_e * n_e, dim=1)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
        reg = (1/2) * (u_e.norm(2).pow(2) + p_e.norm(2).pow(2) + n_e.norm(2).pow(2)) / users.shape[0]
        return bpr_loss + 1e-4 * reg
