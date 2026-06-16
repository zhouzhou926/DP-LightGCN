import os
import yaml
import torch
import numpy as np
import random

class Config:
    def __init__(self, config_path="./config/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self._cfg = yaml.safe_load(f)
        
        # 转换为属性
        for k, v in self._cfg.items():
            setattr(self, k, v)
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 种子
        self._set_seed(self.seed)
    
    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def get_dataset_path(self):
        """获取数据集路径"""
        return os.path.join(self.data_path, self.dataset)
    
    def __repr__(self):
        return f"Config(dataset={self.dataset}, emb_dim={self.emb_dim}, n_layers={self.n_layers}, dp={self.enable_dp})"
