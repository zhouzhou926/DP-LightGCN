"""
配置管理模块
支持多数据集一键切换 + 多种子实验配置
"""
import os
import yaml
import torch
import numpy as np
import random

class Config:
    def __init__(self, config_path="./config/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self._cfg = yaml.safe_load(f)

        # 数据集选择
        self.dataset = self._cfg["dataset"]
        data_paths = self._cfg["data"]
        if self.dataset not in data_paths:
            raise ValueError(f"Unknown dataset: {self.dataset}, options: {list(data_paths.keys())}")
        self.data_path = data_paths[self.dataset]

        # 训练超参数
        self.batch_size = self._cfg["batch_size"]
        self.lr = self._cfg["lr"]
        self.emb_dim = self._cfg["emb_dim"]
        self.n_layers = self._cfg["n_layers"]
        self.epochs = self._cfg["epochs"]
        self.eval_interval = self._cfg["eval_interval"]
        self.early_stop_patience = self._cfg["early_stop_patience"]
        self.decay = self._cfg["decay"]

        # DP参数
        self.delta = float(self._cfg["delta"])
        self.epsilons = self._cfg["epsilons"]
        self.strategies = self._cfg["strategies"]

        # 实验设置
        self.n_seeds = self._cfg["n_seeds"]
        self.seed_base = self._cfg["seed_base"]
        self.top_k = self._cfg["top_k"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Config] Dataset: {self.dataset} | Path: {self.data_path}")
        print(f"[Config] Device: {self.device} | Epsilons: {self.epsilons} | Strategies: {self.strategies}")
        print(f"[Config] Seeds: {self.n_seeds} (base={self.seed_base}) | Dim={self.emb_dim} | Layers={self.n_layers}")

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __repr__(self):
        return f"Config(dataset={self.dataset}, dim={self.emb_dim}, layers={self.n_layers}, seed_base={self.seed_base})"
