"""
DPLightGCN: 差分隐私LightGCN模型
"""
import torch
import torch.nn as nn
import numpy as np
from math import log

class EmbeddingPerturbation:
    def __init__(self, epsilon, delta, budget_strategy="adaptive"):
        self.epsilon = epsilon
        self.delta = delta
        self.budget_strategy = budget_strategy

    def compute_layer_budgets(self, K, embeddings=None):
        if self.budget_strategy == "uniform":
            return [self.epsilon / K] * K
        elif self.budget_strategy == "decreasing":
            weights = [1.0 / (i + 1) for i in range(1, K + 1)]
            total = sum(weights)
            return [self.epsilon * w / total for w in weights]
        elif self.budget_strategy == "adaptive":
            if embeddings is None or any(e is None for e in embeddings):
                return [self.epsilon / K] * K
            norms = [torch.norm(e.detach(), p=2, dim=1).mean().item() for e in embeddings]
            norm_sum = sum(norms) + 1e-8
            weights = [n / norm_sum for n in norms]
            return [self.epsilon * w for w in weights]
        else:
            raise ValueError(f"Unknown: {self.budget_strategy}")

    def get_clip_threshold(self, embedding):
        avg_norm = torch.norm(embedding.detach(), p=2, dim=1).mean().item()
        return max(avg_norm * 2.0, 0.1)

    def clip_and_noise(self, embedding, eps_layer):
        clip_C = self.get_clip_threshold(embedding)
        norms = torch.norm(embedding, p=2, dim=1, keepdim=True)
        scaling = torch.clamp(clip_C / (norms + 1e-8), max=1.0)
        clipped = embedding * scaling
        sigma = clip_C * np.sqrt(2 * np.log(1.25 / self.delta)) / (eps_layer + 1e-8)
        noise = torch.normal(0, sigma, size=clipped.shape, device=clipped.device)
        return clipped + noise

class RenyiAccountant:
    def __init__(self, delta=1e-5):
        self.delta = delta
        self.rdp_orders = [alpha for alpha in range(2, 64, 2)]
    def gaussian_rdp(self, sigma, order):
        return order / (2 * sigma ** 2)
    def compute_total_epsilon(self, sigmas):
        best_eps = float("inf")
        for alpha in self.rdp_orders:
            rdp_sum = sum([self.gaussian_rdp(s, alpha) for s in sigmas])
            eps = rdp_sum + log(1.0 / self.delta) / (alpha - 1)
            if eps < best_eps:
                best_eps = eps
        return best_eps

class DPLightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, n_layers,
                 epsilon=10.0, delta=1e-5, budget_strategy="adaptive"):
        super().__init__()
        self.n_users = n_users; self.n_items = n_items
        self.n_layers = n_layers; self.emb_dim = emb_dim
        self.epsilon = epsilon; self.delta = delta
        self.budget_strategy = budget_strategy
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        self.perturbator = EmbeddingPerturbation(
            epsilon=epsilon, delta=delta, budget_strategy=budget_strategy)

    def forward(self, adj_matrix, apply_dp=True):
        ego = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_emb = [ego]
        layer_emb = []; tmp = ego
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
        all_emb = self.forward(adj_matrix, apply_dp=apply_dp)
        u_e = all_emb[users]; p_e = all_emb[self.n_users + pos_items]
        n_e = all_emb[self.n_users + neg_items]
        pos_s = torch.sum(u_e * p_e, dim=1); neg_s = torch.sum(u_e * n_e, dim=1)
        loss = -torch.mean(torch.log(torch.sigmoid(pos_s - neg_s) + 1e-8))
        reg = (1/2)*(u_e.norm(2).pow(2)+p_e.norm(2).pow(2)+n_e.norm(2).pow(2))/users.shape[0]
        return loss + 1e-4 * reg
