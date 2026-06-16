"""
合成数据集生成器
用于在没有真实数据集时快速验证代码可运行
"""
import os
import numpy as np

def generate_synthetic_data(data_path, n_users=1000, n_items=2000, n_interactions=20000):
    """生成合成LightGCN格式数据
    Args:
        data_path: 输出目录
        n_users: 用户数
        n_items: 物品数
        n_interactions: 交互数（训练集）
    """
    os.makedirs(data_path, exist_ok=True)
    
    np.random.seed(2024)
    
    # 生成用户-物品交互
    users = np.random.randint(0, n_users, n_interactions)
    items = np.random.randint(0, n_items, n_interactions)
    
    # 去重并收集每个用户的交互
    from collections import defaultdict
    user_items = defaultdict(set)
    for u, i in zip(users, items):
        user_items[u].add(i)
    
    # 写入train.txt（每个用户至少2个交互）
    train_path = os.path.join(data_path, "train.txt")
    test_path = os.path.join(data_path, "test.txt")
    
    with open(train_path, 'w') as ftrain, open(test_path, 'w') as ftest:
        for u in range(n_users):
            items = list(user_items.get(u, set()))
            if len(items) < 2:
                # 用户交互太少，补充
                while len(items) < 3:
                    items.append(np.random.randint(0, n_items))
                items = items[:3]
            
            # 前n-1个训练，最后1个测试
            train_str = " ".join([str(u)] + [str(i) for i in items[:-1]])
            test_str = " ".join([str(u)] + [str(items[-1])])
            ftrain.write(train_str + "\n")
            ftest.write(test_str + "\n")
    
    print(f"Generated synthetic data at {data_path}")
    print(f"  Users: {n_users}, Items: {n_items}")
    print(f"  Training interactions: {sum(len(list(v)) for v in user_items.values())}")
    return data_path

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "./data/synthetic"
    generate_synthetic_data(data_path)
