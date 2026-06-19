
import sys; sys.path.insert(0, '.')
import time, torch, os, pandas as pd
from utils.config import Config; from utils.data_loader import LightGCNDataset, BatchSampler
from utils.metrics import get_metrics; from models.dp_mechanism import DPLightGCN

cfg = Config()
ds = LightGCNDataset(cfg.get_dataset_path())
sampler = BatchSampler(ds)
adj = ds.get_adj_tensor(cfg.device)
n_iters = max(1, len(ds)//2048)
print(f'Gowalla: {ds.n_users} users, {n_iters} batches/epoch', flush=True)

results = []
for K in [1, 2, 3, 4]:
    t0 = time.time()
    model = DPLightGCN(ds.n_users, ds.n_items, 64, K, epsilon=5.0, delta=1e-5, budget_strategy='adaptive').to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for ep in range(40):
        model.train()
        for _ in range(n_iters):
            u,p,n = sampler.sample_batch(2048)
            opt.zero_grad()
            l = model.bpr_loss(u.to(cfg.device), p.to(cfg.device), n.to(cfg.device), adj, apply_dp=True)
            l.backward(); opt.step()
    r = get_metrics(model, ds, sampler, top_k_list=[20], device=cfg.device)
    results.append({'K': K, 'Recall@20': r['Recall@20'], 'NDCG@20': r['NDCG@20']})
    print(f'K={K} | R@20={r["Recall@20"]:.4f} | N@20={r["NDCG@20"]:.4f} | {time.time()-t0:.0f}s', flush=True)

os.makedirs('results', exist_ok=True)
pd.DataFrame(results).to_csv('results/ablation_layers.csv', index=False)
print('\nAblation 2: Layers done!', flush=True)
print(pd.DataFrame(results).to_string(), flush=True)
