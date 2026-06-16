# DP-LightGCN

Differentially Private LightGCN with Layer-wise Adaptive Budget Allocation.

## 环境
- Python 3.9~3.11, PyTorch 2.0+, CUDA 11.7+

## 安装
```
pip install torch numpy scipy matplotlib pandas pyyaml
```

## 数据
下载 LightGCN 标准数据集 (Yelp2018/Gowalla/Amazon-Book) 放入 `./data/` 目录。

## 运行
- 完整对比实验: `python train.py`
- 单次实验: `python train.py --epsilon 10 --strategy adaptive`
- 无DP基线: `python train.py --no-dp`

## 结构
- `models/dp_mechanism.py` - DPLightGCN模型 + DP加噪 + RDP会计（核心代码）
- `utils/` - 数据加载、指标、可视化、配置
- `config/config.yaml` - 超参数配置
- `train.py` - 主训练脚本

## 创新点
1. 层级自适应预算分配（基于嵌入范数）
2. 嵌入级高斯扰动
3. Renyi DP Composition 会计
