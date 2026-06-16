"""
合成数据集生成器
用于在没有真实数据集时快速验证代码可运行

用法: python utils/generate_synthetic_data.py [n_users] [n_items] [n_interactions]
默认: 1000用户, 2000物品, 20000交互
"""
import os
import sys
import numpy as np
from collections import defaultdict

def generate_synthetic_data(data_path, n_users=1000, n_items=2000, n_interactions=20000):
    os.makedirs(data_path, exist_ok=True)
    np.random.seed(2024)

    users = np.random.randint(0, n_users, n_interactions)
    items = np.random.randint(0, n_items, n_interactions)

    user_items = defaultdict(set)
    for u, i in zip(users, items):
        user_items[u].add(i)

    train_path = os.path.join(data_path, "train.txt")
    test_path = os.path.join(data_path, "test.txt")

    with open(train_path, 'w') as ftrain, open(test_path, 'w') as ftest:
        for u in range(n_users):
            items_list = list(user_items.get(u, set()))
            if len(items_list) < 2:
                while len(items_list) < 3:
                    items_list.append(np.random.randint(0, n_items))
                items_list = items_list[:3]

            train_str = " ".join([str(u)] + [str(i) for i in items_list[:-1]])
            test_str = " ".join([str(u)] + [str(items_list[-1])])
            ftrain.write(train_str + "\n")
            ftest.write(test_str + "\n")

    print(f"Generated synthetic data at {data_path}")
    print(f"  Users: {n_users}, Items: {n_items}")
    total_inter = sum(len(v) for v in user_items.values())
    print(f"  Training interactions: {total_inter}")


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:3]] if len(sys.argv) > 2 else []
    kwargs = {}
    if len(args) >= 1:
        kwargs['n_users'] = args[0]
    if len(args) >= 2:
        kwargs['n_items'] = args[1]
    if len(sys.argv) >= 4:
        kwargs['n_interactions'] = int(sys.argv[3])
    generate_synthetic_data("./data/synthetic", **kwargs)
