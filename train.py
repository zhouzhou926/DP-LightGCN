"""
DPLightGCN 训练主脚本
"""
import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
from utils.data_loader import LightGCNDataset, BatchSampler
from utils.metrics import get_metrics, print_metrics
from utils.visualize import save_results_csv
from models.dp_mechanism import DPLightGCN


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        print(f"Device: {self.device}")
        print("Loading dataset...")
        self.dataset = LightGCNDataset(config.get_dataset_path())
        self.sampler = BatchSampler(self.dataset)
        print(f"  Users: {self.dataset.n_users}, Items: {self.dataset.n_items}")
        n_inter = sum(len(items) for _, items in self.dataset.train_data)
        print(f"  Interactions: {n_inter}")

    def train_epoch(self, model, optimizer, apply_dp=True):
        model.train()
        total_loss = 0
        n_batches = 0
        adj_matrix = self.dataset.get_adj_tensor(self.device)
        n_train = len(self.dataset)
        n_iters = max(1, n_train // self.config.batch_size)
        for _ in range(n_iters):
            users, pos_items, neg_items = self.sampler.sample_batch(self.config.batch_size)
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

    def evaluate(self, model, apply_dp=True):
        # 评估时不加噪（加噪只在训练时）
        return get_metrics(model, self.dataset, self.sampler, top_k_list=[20], device=self.device)

    def run_experiment(self, epsilon, budget_strategy, enable_dp=True):
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"Experiment: epsilon={epsilon}, strategy={budget_strategy}, dp={enable_dp}")
        print(f"{sep}")
        model = DPLightGCN(
            self.dataset.n_users, self.dataset.n_items,
            self.config.emb_dim, self.config.n_layers,
            epsilon=epsilon, delta=self.config.delta,
            budget_strategy=budget_strategy,
        ).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=0)
        best_recall = -1
        patience_counter = 0
        for epoch in range(1, self.config.epochs + 1):
            t_start = time.time()
            loss = self.train_epoch(model, optimizer, apply_dp=enable_dp)
            if epoch % self.config.eval_interval == 0:
                results = self.evaluate(model)
                recall = results.get("Recall@20", 0)
                ndcg = results.get("NDCG@20", 0)
                t_elapsed = time.time() - t_start
                print(f"  Epoch {epoch:4d} | loss={loss:.4f} | Recall@20={recall:.4f} | NDCG@20={ndcg:.4f} | {t_elapsed:.1f}s")
                if recall > best_recall:
                    best_recall = recall
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stop_patience:
                        print(f"  Early stop at epoch {epoch}")
                        break
        final_results = self.evaluate(model)
        print(f"\nFinal results:")
        print_metrics(final_results)
        return {"results": final_results, "best_recall": best_recall}

    def run_full_comparison(self):
        epsilons = self.config.epsilon if isinstance(self.config.epsilon, list) else [self.config.epsilon]
        strategies = ["uniform", "decreasing", "adaptive"] if self.config.enable_dp else ["uniform"]
        all_results = {}
        print(">>> Running baseline: LightGCN (No DP) <<<")
        baseline = self.run_experiment(epsilon=1e8, budget_strategy="uniform", enable_dp=False)
        all_results["LightGCN(NoDP)"] = {1e8: baseline["results"]}
        for strategy in strategies:
            all_results[strategy] = {}
            for eps in epsilons:
                result = self.run_experiment(epsilon=eps, budget_strategy=strategy, enable_dp=True)
                all_results[strategy][eps] = result["results"]
        save_results_csv(all_results, save_path="./results/experiment_results.csv")
        print("\nAll experiments completed!")
        return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--no-dp", action="store_true")
    args = parser.parse_args()
    cfg = Config(args.config)
    if args.epsilon is not None:
        cfg.epsilon = args.epsilon
    if args.strategy is not None:
        cfg.budget_strategy = args.strategy
    if args.no_dp:
        cfg.enable_dp = False
    trainer = Trainer(cfg)
    if args.epsilon is not None or args.no_dp:
        trainer.run_experiment(
            epsilon=args.epsilon or 1e8,
            budget_strategy=args.strategy or cfg.budget_strategy,
            enable_dp=not args.no_dp,
        )
    else:
        trainer.run_full_comparison()

if __name__ == "__main__":
    main()
