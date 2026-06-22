
"""
联邦学习数据划分模块
"""
import numpy as np
from collections import defaultdict

class FederatedDataPartitioner:
    def __init__(self, dataset, n_clients=10, partition='uniform', seed=2024):
        self.dataset = dataset
        self.n_clients = n_clients
        self.partition = partition
        np.random.seed(seed)
        self.client_data = self._partition()

    def _partition(self):
        n_users = self.dataset.n_users
        if self.partition == 'uniform':
            users_per_client = n_users // self.n_clients
            indices = np.random.permutation(n_users)
            client_users = {}
            for c in range(self.n_clients):
                start = c * users_per_client
                end = start + users_per_client if c < self.n_clients - 1 else n_users
                client_users[c] = indices[start:end].tolist()
        elif self.partition == 'dirichlet':
            from numpy.random import dirichlet
            alpha = [0.5] * self.n_clients
            proportions = dirichlet(alpha, size=n_users)
            assignments = np.argmax(proportions, axis=1)
            client_users = defaultdict(list)
            for u, c in enumerate(assignments):
                client_users[int(c)].append(u)
        else:
            raise ValueError(f'Unknown partition: {self.partition}')
        
        client_data = {}
        for c, users in client_users.items():
            user_set = set(users)
            local_train = [(u, items) for u, items in self.dataset.train_data if u in user_set]
            local_test = [(u, items) for u, items in self.dataset.test_data if u in user_set]
            client_data[c] = {
                'users': users,
                'n_users': len(users),
                'train_data': local_train,
                'test_data': local_test,
            }
        return client_data
    
    def get_client_info(self):
        info = {}
        for c, data in self.client_data.items():
            n_train = sum(len(items) for _, items in data['train_data'])
            info[c] = {
                'n_users': data['n_users'],
                'n_train': n_train,
                'avg_inter': n_train / max(data['n_users'], 1),
            }
        return info

    def print_summary(self):
        info = self.get_client_info()
        n_users = sum(v['n_users'] for v in info.values())
        n_train = sum(v['n_train'] for v in info.values())
        print(f'Clients: {self.n_clients} | Total users: {n_users} | Total inter: {n_train}')
        print(f'Partition: {self.partition}')
        for c, v in info.items():
            print(f'  Client {c:2d}: users={v["n_users"]:4d}, inter={v["n_train"]:6d}, avg={v["avg_inter"]:.1f}')
