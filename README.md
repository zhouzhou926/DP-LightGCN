# DP-LightGCN

Differentially Private LightGCN with Layer-wise Adaptive Budget Allocation.

## 环境
- Python 3.9~3.11, PyTorch 2.0+, CUDA 11.7+

## 安装
```bash
pip install torch numpy scipy matplotlib pandas pyyaml
```

## 数据

### 已下载
- **Gowalla**: 已就绪 (29,858 users, 4.4MB)
- **Amazon-Book**: 已就绪 (52,643 users, 13.5MB)

### 需手动下载
- **Yelp2018**: 所有公开链接已失效，请从以下任一方式获取:
  1. Kaggle搜索 "lightgcn yelp2018" 下载 train.txt / test.txt
  2. 使用 `utils/build_dataset.py` 从Yelp开放数据集原始JSON构建
  3. 联系论文原作者索取

### 数据集格式
每行: `user_id item_id_1 item_id_2 ...`
用户和物品ID需从0开始连续编号。

## 运行
```bash
# 修改 config/config.yaml 中 data_path 指向对应数据集目录
# Gowalla 示例:
data_path: "./data/gowalla"

# 完整对比实验
python train.py

# 单次DP实验
python train.py --epsilon 10 --strategy adaptive

# 无DP基线
python train.py --no-dp
```

## 项目结构
- `models/dp_mechanism.py` - DPLightGCN模型 + DP加噪 + RDP会计（核心代码）
- `utils/` - 数据加载、指标、可视化、配置、数据集构建
- `config/config.yaml` - 超参数配置
- `train.py` - 主训练脚本
- `utils/download_datasets.py` - 数据集下载脚本

## 创新点
1. 层级自适应预算分配（基于嵌入范数）
2. 嵌入级高斯扰动
3. Renyi DP Composition 会计
