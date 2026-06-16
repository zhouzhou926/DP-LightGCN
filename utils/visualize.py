"""
可视化模块：隐私-效用曲线、对比柱状图等
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 150

SAVE_DIR = "./results/figures"


def plot_privacy_utility_tradeoff(results_dict, save_path=None):
    """绘制隐私-效用曲线
    results_dict: {budget_strategy: {epsilon: {Recall@20: ..., NDCG@20: ...}}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    markers = {"uniform": "o", "decreasing": "s", "adaptive": "^"}
    colors = {"uniform": "#E74C3C", "decreasing": "#3498DB", "adaptive": "#2ECC71"}

    for strategy, results in results_dict.items():
        epsilons = sorted(results.keys())
        recalls = [results[eps].get("Recall@20", 0) for eps in epsilons]
        ndcgs = [results[eps].get("NDCG@20", 0) for eps in epsilons]

        axes[0].plot(epsilons, recalls, marker=markers[strategy],
                     color=colors[strategy], label=strategy, linewidth=2, markersize=6)
        axes[1].plot(epsilons, ndcgs, marker=markers[strategy],
                     color=colors[strategy], label=strategy, linewidth=2, markersize=6)

    axes[0].set_xlabel("Epsilon (privacy budget)")
    axes[0].set_ylabel("Recall@20")
    axes[0].set_title("Privacy-Utility Trade-off (Recall@20)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epsilon (privacy budget)")
    axes[1].set_ylabel("NDCG@20")
    axes[1].set_title("Privacy-Utility Trade-off (NDCG@20)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()


def plot_comparison_bar(results_df, save_path=None):
    """绘制不同方法的对比柱状图
    results_df: DataFrame, columns=[Model, Epsilon, Recall@20, NDCG@20]
    """
    epsilons = results_df["Epsilon"].unique()
    n_eps = len(epsilons)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, metric in enumerate(["Recall@20", "NDCG@20"]):
        ax = axes[idx]
        bar_width = 0.12
        models = results_df["Model"].unique()
        x = np.arange(n_eps)

        for i, model in enumerate(models):
            values = results_df[results_df["Model"] == model][metric].values
            offset = (i - len(models) / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, values, bar_width, label=model)

        ax.set_xlabel("Epsilon")
        ax.set_ylabel(metric)
        ax.set_title(f"Comparison: {metric}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(e) for e in epsilons])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()


def save_results_csv(all_results, save_path="./results/experiment_results.csv"):
    """保存实验结果到CSV"""
    rows = []
    for strategy, eps_results in all_results.items():
        for eps, metrics in eps_results.items():
            rows.append({
                "Strategy": strategy,
                "Epsilon": eps,
                **metrics,
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")
    return df
