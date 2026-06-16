"""
LightGCN 标准数据集构建脚本
=============================
从原始评分文件构建LightGCN所需的 train.txt / test.txt 格式。

用法:
  1. 从原始数据源下载数据集（详见下方说明）
  2. 运行: python utils/build_dataset.py --name yelp2018 --input <原始文件路径>

数据源说明:
----------------
Yelp2018:
  原始数据: https://www.yelp.com/dataset (Yelp Open Dataset, 需申请)
  处理后格式: user_id business_id rating timestamp (tsv/csv)

Gowalla:
  原始数据: https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz
  处理后格式: user POI location check-in count

Amazon-Book:
  原始数据: https://jmcauley.ucsd.edu/data/amazon/ (5-core)
  处理后格式: user_id item_id rating timestamp (tsv/csv)

备用方案（推荐）:
-----------------
如果不想从原始数据构建，可以在Kaggle上搜索预处理好的LightGCN格式数据:
  https://www.kaggle.com/datasets?search=lightgcn
搜索关键词: "LightGCN yelp2018" 或 "LightGCN gowalla"

输出格式:
----------------
train.txt: 每行: user item1 item2 item3 ... (前80%交互)
test.txt:  每行: user test_item (留一法，最后1个交互)
"""
import os
import sys
import argparse
import random
from collections import defaultdict


def load_interactions(filepath, sep="\t", has_header=True, user_col=0, item_col=1):
    """从原始评分文件加载交互"""
    user_items = defaultdict(set)
    with open(filepath, 'r', encoding='utf-8') as f:
        if has_header:
            next(f)
        for line in f:
            parts = line.strip().split(sep)
            if len(parts) <= max(user_col, item_col):
                continue
            try:
                u = int(parts[user_col])
                i = int(parts[item_col])
                user_items[u].add(i)
            except ValueError:
                continue
    return user_items


def reindex(user_items):
    """重索引用户和物品ID为0~N-1"""
    all_users = sorted(user_items.keys())
    all_items = sorted(set().union(*user_items.values()))
    user_map = {old: new for new, old in enumerate(all_users)}
    item_map = {old: new for new, old in enumerate(all_items)}
    
    reindexed = defaultdict(set)
    for u, items in user_items.items():
        reindexed[user_map[u]] = {item_map[i] for i in items}
    return reindexed, len(all_users), len(all_items)


def split_and_save(user_items, n_users, n_items, output_dir, test_per_user=1):
    """划分训练/测试集并保存为LightGCN格式"""
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.txt")
    test_path = os.path.join(output_dir, "test.txt")
    
    with open(train_path, 'w') as ftrain, open(test_path, 'w') as ftest:
        for u in range(n_users):
            items = list(user_items.get(u, set()))
            if len(items) < 2:
                continue
            
            # 随机打乱
            random.shuffle(items)
            
            # 按留一法划分（最后一1个测试，其余训练）
            n_test = min(test_per_user, len(items) - 1)
            test_items = items[:n_test]
            train_items = items[n_test:]
            
            if train_items:
                ftrain.write(str(u) + " " + " ".join(str(i) for i in train_items) + "\n")
            if test_items:
                ftest.write(str(u) + " " + " ".join(str(i) for i in test_items) + "\n")
    
    print(f"[Build] Dataset saved to {output_dir}")
    print(f"  Users: {n_users}, Items: {n_items}")
    print(f"  Interactions: {sum(len(v) for v in user_items.values())}")
    print(f"  Train file: {train_path}")
    print(f"  Test file: {test_path}")


def build_from_raw(input_file, output_dir, sep="\t", has_header=True, 
                   user_col=0, item_col=1, min_interactions=5):
    """从原始评分文件构建LightGCN数据集（5-core过滤）"""
    print(f"[Build] Loading raw data from {input_file}")
    user_items = load_interactions(input_file, sep=sep, has_header=has_header,
                                    user_col=user_col, item_col=item_col)
    print(f"[Build] Raw interactions: {sum(len(v) for v in user_items.values())}")
    
    # 5-core过滤：每个用户至少min_interactions个交互
    user_items = {u: items for u, items in user_items.items() 
                  if len(items) >= min_interactions}
    print(f"[Build] After {min_interactions}-core filtering: {len(user_items)} users")
    
    user_items, n_users, n_items = reindex(user_items)
    split_and_save(user_items, n_users, n_items, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Build LightGCN dataset from raw data")
    parser.add_argument("--input", type=str, required=True, help="Raw input file path")
    parser.add_argument("--output", type=str, default="./data/my_dataset", help="Output directory")
    parser.add_argument("--sep", type=str, default="\t", help="Separator (default: tab)")
    parser.add_argument("--no-header", action="store_true", help="Input file has no header")
    parser.add_argument("--user-col", type=int, default=0, help="User column index (0-based)")
    parser.add_argument("--item-col", type=int, default=1, help="Item column index (0-based)")
    parser.add_argument("--min-inter", type=int, default=5, help="Minimum interactions per user")
    args = parser.parse_args()
    
    build_from_raw(
        args.input, args.output,
        sep=args.sep,
        has_header=not args.no_header,
        user_col=args.user_col,
        item_col=args.item_col,
        min_interactions=args.min_inter,
    )


if __name__ == "__main__":
    main()
