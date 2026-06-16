"""
数据加载模块
支持LightGCN标准数据格式：
- train.txt: 训练集，每行"user_id item_id_1 item_id_2 ..."
- test.txt: 测试集（留一法，最后一项为测试项，其余为验证集）
- user_list.txt / item_list.txt: 用户/物品列表
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader

class LightGCNDataset(Dataset):
    """LightGCN格式的数据集"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.train_data, self.test_data = self._load_data()
        self.n_users, self.n_items = self._get_statics()
        self.adj_matrix = self._build_adj_matrix()
    
    def _load_data(self):
        """加载train.txt和test.txt"""
        train_data = []
        test_data = []
        
        # 读取训练集
        with open(os.path.join(self.data_path, "train.txt"), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                user = int(parts[0])
                items = [int(x) for x in parts[1:]]
                train_data.append((user, items))
        
        # 读取测试集
        with open(os.path.join(self.data_path, "test.txt"), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                user = int(parts[0])
                items = [int(x) for x in parts[1:]]
                test_data.append((user, items))
        
        return train_data, test_data
    
    def _get_statics(self):
        """获取用户数和物品数"""
        n_users = max([u for u, _ in self.train_data] + [u for u, _ in self.test_data]) + 1
        n_items = max(
            [max(items) for _, items in self.train_data if items] +
            [max(items) for _, items in self.test_data if items]
        ) + 1
        return n_users, n_items
    
    def _build_adj_matrix(self):
        """构建归一化的邻接矩阵 (n_users+n_items) x (n_users+n_items)"""
        n_nodes = self.n_users + self.n_items
        
        rows, cols = [], []
        for user, items in self.train_data:
            for item in items:
                rows.append(user)
                cols.append(self.n_users + item)
                rows.append(self.n_users + item)
                cols.append(user)
        
        data = np.ones(len(rows))
        adj = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        
        # 对称归一化: D^{-1/2} A D^{-1/2}
        row_sum = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(row_sum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        return norm_adj.tocoo()
    
    def get_adj_tensor(self, device):
        """返回PyTorch稀疏张量"""
        indices = torch.LongTensor([self.adj_matrix.row, self.adj_matrix.col])
        values = torch.FloatTensor(self.adj_matrix.data)
        shape = torch.Size(self.adj_matrix.shape)
        return torch.sparse.FloatTensor(indices, values, shape).to(device)
    
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, idx):
        user, pos_items = self.train_data[idx]
        return user, pos_items


class BatchSampler:
    """为LightGCN定制的批量采样器
    
    从训练集中采样batch_size个(user, pos_item, neg_item)三元组
    """
    
    def __init__(self, dataset, n_neg=1):
        self.dataset = dataset
        self.n_neg = n_neg
        self.users = [u for u, items in dataset.train_data]
        self.pos_items = [items for _, items in dataset.train_data]
        
        # 每个用户的正向物品集合（用于负采样过滤）
        self.user_pos_dict = {}
        for u, items in dataset.train_data:
            self.user_pos_dict[u] = set(items)
    
    def sample_batch(self, batch_size):
        """采样一个batch"""
        batch_users = np.random.choice(self.users, batch_size, replace=True)
        
        pos_items = []
        neg_items = []
        
        for u in batch_users:
            # 采样一个正向物品
            pos = np.random.choice(list(self.user_pos_dict[u]))
            pos_items.append(pos)
            
            # 采样一个负向物品（不在正向集中）
            neg = np.random.randint(0, self.dataset.n_items)
            while neg in self.user_pos_dict[u]:
                neg = np.random.randint(0, self.dataset.n_items)
            neg_items.append(neg)
        
        return (
            torch.LongTensor(batch_users),
            torch.LongTensor(pos_items),
            torch.LongTensor(neg_items)
        )
