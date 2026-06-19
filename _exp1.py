
import sys; sys.path.insert(0, '.')
import time, torch, os, pandas as pd
from utils.config import Config
from utils.data_loader import LightGCNDataset, BatchSampler
from utils.metrics import get_metrics
from models.dp_mechanism import DPLightGCN

cfg = Config()
ds = LightGCNDataset(cfg.get_dataset_path())
sampler = BatchSampler(ds)
adj = ds.get_adj_tensor(cfg.device)
n_iters = max(1, len(ds)//2048)
print(f"Gowalla: {ds.n_users} users, {n_iters} batches/epoch", flush=True)

results = []
for eps in [5, 10, 20]:
    for use_clip in [True, False]:
        label = "Adaptive+Clip" if use_clip else "Adaptive(NoClip)"
        t0 = time.time()
        model = DPLightGCN(ds.n_users, ds.n_items, 64, 3, epsilon=eps, delta=1e-5, budget_strategy="adaptive").to(cfg.device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        for ep in range(40):
            model.train()
            for _ in range(n_iters):
                u,p,n = sampler.sample_batch(2048)
                opt.zero_grad()
                l = model.bpr_loss(u.to(cfg.device), p.to(cfg.device), n.to(cfg.device), adj, apply_dp=use_clip)
                l.backward(); opt.step()
        r = get_metrics(model, ds, sampler, top_k_list=[20], device=cfg.device)
        t = time.time() - t0
        results.append({"Model": label, "Epsilon": eps, "Recall@20": r["Recall@20"], "NDCG@20": r["NDCG@20"]})
        print(f"{label:20s} eps={eps:2d} | R@20={r['Recall@20']:.4f} | N@20={r['NDCG@20']:.4f} | {t:.0f}s", flush=True)

# Save
os.makedirs("results", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("results/ablation_clipping.csv", index=False)
print("\n=== Ablation 1: Clipping Strategy ===", flush=True)
print(df.to_string(), flush=True)
