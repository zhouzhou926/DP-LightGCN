
"""
DP-MF: 差分隐私矩阵分解基线模型
使用DP-SGD风格的梯度裁剪+噪声
"""
import torch
import torch.nn as nn
import numpy as np

class DPMF(nn.Module):
    """差分隐私矩阵分解"""
    def __init__(self, n_users, n_items, emb_dim=64, 
                 epsilon=10.0, delta=1e-5, l2_reg=1e-4):
        super().__init__()
        self.n_users = n_users; self.n_items = n_items
        self.emb_dim = emb_dim; self.epsilon = epsilon
        self.delta = delta; self.l2_reg = l2_reg
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        # 梯度裁剪阈值
        self.clip_C = 1.0

    def forward(self):
        return self.user_embedding.weight, self.item_embedding.weight

    def bpr_loss(self, users, pos_items, neg_items):
        u = self.user_embedding(users)
        p = self.item_embedding(pos_items)
        n = self.item_embedding(neg_items)
        pos_s = torch.sum(u * p, dim=1)
        neg_s = torch.sum(u * n, dim=1)
        loss = -torch.mean(torch.log(torch.sigmoid(pos_s - neg_s) + 1e-8))
        reg = (1/2)*(u.norm(2).pow(2)+p.norm(2).pow(2)+n.norm(2).pow(2))/users.shape[0]
        return loss + self.l2_reg * reg

    def clip_gradients(self):
        """逐用户裁剪梯度（DP-SGD风格）"""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                # 裁剪单个参数梯度
                norm = p.grad.norm(2)
                total_norm += norm.item() ** 2
                if norm > self.clip_C:
                    p.grad.mul_(self.clip_C / norm)
        return np.sqrt(total_norm)

    def add_noise(self):
        """添加高斯噪声（实现(epsilon,delta)-DP）"""
        sigma = self.clip_C * np.sqrt(2 * np.log(1.25 / self.delta)) / (self.epsilon + 1e-8)
        with torch.no_grad():
            for p in self.parameters():
                if p.grad is not None:
                    noise = torch.normal(0, sigma, size=p.grad.shape, device=p.grad.device)
                    p.grad.add_(noise)
