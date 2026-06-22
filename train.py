"""
DP-LightGCN ?????
?????? + ????? + ?????mean?std?
"""
import os
import sys as _sys
import argparse
import time
import numpy as np
import torch
import torch.optim as optim

_sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
from utils.data_loader import LightGCNDataset, BatchSampler
from utils.metrics import get_metrics, print_metrics
from utils.visualize import save_results_csv
from models.dp_mechanism import DPLightGCN


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        print(f"[Config] Device: {self.device}")

    def load_dataset(self):
        print(f"[Data] Loading dataset from {self.config.data_path}...")
        dataset = LightGCNDataset(self.config.data_path)
        sampler = BatchSampler(dataset)
        n_inter = sum(len(items) for _, items in dataset.train_data)
        print(f"[Data] Users: {dataset.n_users}, Items: {dataset.n_items}, Interactions: {n_inter}")
        return dataset, sampler

    def train_epoch(self, model, optimizer, adj_matrix, dataset, sampler, apply_dp=True):
        model.train()
        total_loss = 0.0
        n_batches = 0
        n_train = len(dataset)
        n_iters = max(1, n_train // self.config.batch_size)
        for _ in range(n_iters):
            users, pos_items, neg_items = sampler.sample_batch(self.config.batch_size)
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)
            optimizer.zero_grad()
            loss = model.bpr_loss(users, pos_items, neg_items, adj_matrix, apply_dp=apply_dp)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(1, n_batches)

    def evaluate(self, model, dataset, sampler):
        return get_metrics(model, dataset, sampler, top_k_list=[self.config.top_k], device=self.device)

    def run_experiment(self, epsilon, budget_strategy, enable_dp=True, seed=2024):
        self.config.set_seed(seed)
        dataset, sampler = self.load_dataset()
        adj_matrix = dataset.get_adj_tensor(self.device)

        if enable_dp:
            label = f"DP-LightGCN | eps={epsilon} | {budget_strategy} | seed={seed}"
        else:
            label = f"LightGCN (No DP) | seed={seed}"

        sep = "=" * 60
        print()
        print(sep)
        print(f"[Experiment] {label}")
        print(sep)

        model = DPLightGCN(
            dataset.n_users, dataset.n_items,
            self.config.emb_dim, self.config.n_layers,
            epsilon=epsilon, delta=self.config.delta,
            budget_strategy=budget_strategy,
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=0)
        best_recall = -1.0
        best_epoch = -1
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            t_start = time.time()
            loss = self.train_epoch(model, optimizer, adj_matrix, dataset, sampler, apply_dp=enable_dp)
            if epoch % self.config.eval_interval == 0:
                results = self.evaluate(model, dataset, sampler)
                recall = results.get(f"Recall@{self.config.top_k}", 0.0)
                ndcg = results.get(f"NDCG@{self.config.top_k}", 0.0)
                t_cost = time.time() - t_start
                print(f"  Epoch {epoch:4d} | loss={loss:.4f} | R{self.config.top_k}={recall:.4f} | N{self.config.top_k}={ndcg:.4f} | {t_cost:.1f}s")
                if recall > best_recall:
                    best_recall = recall
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stop_patience:
                        print(f"  >> Early stopped at epoch {epoch}")
                        break

        final_results = self.evaluate(model, dataset, sampler)
        print(f"[Result] Best epoch: {best_epoch}, Best Recall@{self.config.top_k}: {best_recall:.4f}")
        print("[Result] Final:")
        print_metrics(final_results)
        return {"results": final_results, "best_recall": best_recall, "best_epoch": best_epoch}

    def run_multi_seed(self, epsilon, strategy, enable_dp=True, n_seeds=3, seed_base=2024):
        recalls, ndcgs, best_epochs = [], [], []
        for seed_offset in range(n_seeds):
            seed = seed_base + seed_offset
            result = self.run_experiment(epsilon=epsilon, budget_strategy=strategy, enable_dp=enable_dp, seed=seed)
            metrics = result["results"]
            recalls.append(metrics.get(f"Recall@{self.config.top_k}", 0.0))
            ndcgs.append(metrics.get(f"NDCG@{self.config.top_k}", 0.0))
            best_epochs.append(result["best_epoch"])

        recall_mean = float(np.mean(recalls))
        recall_std = float(np.std(recalls))
        ndcg_mean = float(np.mean(ndcgs))
        ndcg_std = float(np.std(ndcgs))

        K = self.config.top_k
        print(f"[Aggregate] {strategy:12s} | eps={epsilon:>6.1f} | R{K}={recall_mean:.6f}+-{recall_std:.6f} | N{K}={ndcg_mean:.6f}+-{ndcg_std:.6f} | epochs={best_epochs}")
        return {
            "results": {
                f"Recall@{K}": recall_mean,
                f"Recall@{K}_std": recall_std,
                f"NDCG@{K}": ndcg_mean,
                f"NDCG@{K}_std": ndcg_std,
            },
            "seeds": {"recalls": recalls, "ndcgs": ndcgs, "best_epochs": best_epochs}
        }

    def run_complete_experiment(self):
        epsilons = self.config.epsilons
        strategies = self.config.strategies
        n_seeds = self.config.n_seeds
        seed_base = self.config.seed_base
        all_results = {"LightGCN(NoDP)": {}, "uniform": {}, "decreasing": {}, "adaptive": {}}
        sep = "=" * 60

        print()
        print(sep)
        print(f"[Main] >>> LightGCN (No DP) baseline ({n_seeds} seeds) <<<")
        print(sep)
        result = self.run_multi_seed(epsilon=1e8, strategy="uniform", enable_dp=False, n_seeds=n_seeds, seed_base=seed_base)
        all_results["LightGCN(NoDP)"][float("inf")] = result["results"]

        for strategy in strategies:
            all_results[strategy] = {}
            for eps in epsilons:
                print()
                print(sep)
                print(f"[Main] >>> DP-LightGCN | {strategy} | eps={eps} ({n_seeds} seeds) <<<")
                print(sep)
                result = self.run_multi_seed(epsilon=eps, strategy=strategy, enable_dp=True, n_seeds=n_seeds, seed_base=seed_base)
                all_results[strategy][eps] = result["results"]

        csv_path = f"./results/{self.config.dataset}_experiment_results.csv"
        save_results_csv(all_results, save_path=csv_path)
        print(f"[Main] Results saved to {csv_path}")
        self._print_summary_table(all_results)
        return all_results

    def _print_summary_table(self, all_results):
        K = self.config.top_k
        print()
        print("=" * 70)
        print(f"[Summary] Final Results: {self.config.dataset}")
        print("=" * 70)
        print(f"{'Model':<25s} {'eps':>5s}     {'Recall@'+str(K):>16s} {'NDCG@'+str(K):>16s}")
        print("-" * 70)
        for mk in ["LightGCN(NoDP)", "uniform", "decreasing", "adaptive"]:
            if mk not in all_results:
                continue
            for eps in sorted(all_results[mk].keys()):
                m = all_results[mk][eps]
                r = m.get(f"Recall@{K}", 0)
                rs = m.get(f"Recall@{K}_std", 0)
                n = m.get(f"NDCG@{K}", 0)
                ns = m.get(f"NDCG@{K}_std", 0)
                ed = f"{eps:.0f}" if eps not in (float("inf"),) and eps < 1e6 else "inf"
                if rs > 0:
                    print(f"{mk:<25s} {ed:>5s}     {r:.6f}+-{rs:.6f}  {n:.6f}+-{ns:.6f}")
                else:
                    print(f"{mk:<25s} {ed:>5s}     {r:.6f}            {n:.6f}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="DP-LightGCN Training")
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-dp", action="store_true")
    args = parser.parse_args()

    cfg = Config(args.config)
    if args.dataset is not None:
        cfg.dataset = args.dataset
        data_paths = cfg._cfg["data"]
        if cfg.dataset not in data_paths:
            raise ValueError(f"Unknown dataset: {cfg.dataset}, options: {list(data_paths.keys())}")
        cfg.data_path = data_paths[cfg.dataset]

    trainer = Trainer(cfg)
    if args.epsilon is not None or args.no_dp:
        seed = args.seed or cfg.seed_base
        eps = args.epsilon if args.epsilon is not None else 1e8
        strategy = args.strategy or cfg.strategies[0]
        trainer.run_experiment(epsilon=eps, budget_strategy=strategy, enable_dp=not args.no_dp, seed=seed)
    else:
        trainer.run_complete_experiment()


if __name__ == "__main__":
    main()
