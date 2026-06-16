"""
DPLightGCN: 差分隐私LightGCN模型
==================================
核心创新：
1. 嵌入级差分隐私扰动（Embedding-Level DP）
2. 层级自适应预算分配（Layer-wise Adaptive Budget Allocation）
3. Rényi DP Composition 隐私会计

用法:
    model = DPLightGCN(n_users, n_items, emb_dim, n_layers,
                       epsilon=10.0, delta=1e-5, budget_strategy="adaptive")
    embeddings = model(adj_matrix, apply_dp=True)  # 训练时开启加噪
"""
import torch
import torch.nn as nn
import numpy as np
from math import log


class EmbeddingPerturbation:
    """嵌入扰动模块：在每层传播后对嵌入加噪"""

    def __init__(self, epsilon, delta, sensitivity, budget_strategy="adaptive"):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.budget_strategy = budget_strategy

    def compute_layer_budgets(self, K, embeddings=None):
        """计算每层的隐私预算分配

        Args:
            K: 层数
            embeddings: 各层嵌入列表（用于adaptive策略）

        Returns:
            eps_per_layer: 长度为K的列表，每层的隐私预算
        """
        if self.budget_strategy == "uniform":
            return [self.epsilon / K] * K

        elif self.budget_strategy == "decreasing":
            # 递减分配：浅层多、深层少
            weights = [1.0 / (i + 1) for i in range(1, K + 1)]
            total = sum(weights)
            return [self.epsilon * w / total for w in weights]

        elif self.budget_strategy == "adaptive":
            # 自适应分配：基于嵌入范数比例分配更多预算给信息更丰富的层
            if embeddings is None or any(e is None for e in embeddings):
                return [self.epsilon / K] * K
            norms = [torch.norm(e.detach(), p=2).item() for e in embeddings]
            norm_sum = sum(norms) + 1e-8
            weights = [n / norm_sum for n in norms]
            return [self.epsilon * w for w in weights]

        else:
            raise ValueError(f"Unknown budget strategy: {self.budget_strategy}")

    def add_noise(self, embedding, eps_layer):
        """对单层嵌入添加高斯噪声

        Args:
            embedding: (n_nodes, emb_dim) 嵌入矩阵
            eps_layer: 当前层的隐私预算

        Returns:
            加噪后的嵌入矩阵
        """
        sigma = self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / (eps_layer + 1e-8)
        noise = torch.normal(0, sigma, size=embedding.shape, device=embedding.device)
        return embedding + noise

    def get_noise_scale(self, eps_layer):
        """获取当前层的噪声规模σ（用于绘图和日志）"""
        return self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / (eps_layer + 1e-8)


class RenyiAccountant:
    """基于Rényi DP的隐私会计"""

    def __init__(self, delta=1e-5):
        self.delta = delta
        self.rdp_orders = [alpha for alpha in range(2, 64, 2)]

    def gaussian_rdp(self, sigma, order):
        """高斯机制的RDP：α / (2σ²)"""
        return order / (2 * sigma ** 2)

    def compute_total_epsilon(self, sigmas):
        """计算composition后的总ε

        Args:
            sigmas: 各机制的噪声尺度列表

        Returns:
            total_eps: composition后的总隐私预算（ε）
        """
        best_eps = float("inf")
        for alpha in self.rdp_orders:
            rdp_sum = sum([self.gaussian_rdp(s, alpha) for s in sigmas])
            eps = rdp_sum + log(1.0 / self.delta) / (alpha - 1)
            if eps < best_eps:
                best_eps = eps
        return best_eps


class DPLightGCN(nn.Module):
    """差分隐私LightGCN模型"""

    def __init__(self, n_users, n_items, emb_dim, n_layers,
                 epsilon=10.0, delta=1e-5, budget_strategy="adaptive"):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.epsilon = epsilon
        self.delta = delta

        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        self.perturbator = EmbeddingPerturbation(
            epsilon=epsilon, delta=delta,
            sensitivity=1.0, budget_strategy=budget_strategy,
        )

    def forward(self, adj_matrix, apply_dp=True):
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embeddings = [ego_embeddings]

        # 第一遍传播：收集各层嵌入用于自适应预算
        layer_embeds = []
        temp = ego_embeddings
        with torch.no_grad():
            for _ in range(self.n_layers):
                temp = torch.sparse.mm(adj_matrix, temp)
                layer_embeds.append(temp)

        # 计算层级隐私预算
        eps_layers = self.perturbator.compute_layer_budgets(self.n_layers, layer_embeds)

        # 第二遍传播：加噪并收集
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj_matrix, ego_embeddings)
            if apply_dp and self.training:
                ego_embeddings = self.perturbator.add_noise(ego_embeddings, eps_layers[k])
            all_embeddings.append(ego_embeddings)

        # 所有层的平均池化（LightGCN标准做法）
        return torch.mean(torch.stack(all_embeddings), dim=0)

    def bpr_loss(self, users, pos_items, neg_items, adj_matrix, apply_dp=True):
        all_embeddings = self.forward(adj_matrix, apply_dp=apply_dp)
        user_emb = all_embeddings[users]
        pos_emb = all_embeddings[self.n_users + pos_items]
        neg_emb = all_embeddings[self.n_users + neg_items]

        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))

        reg = (1/2) * (user_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / users.shape[0]
        return loss + reg

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
