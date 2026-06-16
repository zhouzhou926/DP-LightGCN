"""
可视化模块：隐私-效用曲线、对比柱状图、训练过程曲线
"""
import os
import matplotlib
matplotlib.use("Agg")  # 无GUI后端，避免依赖显示器
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 150
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def plot_privacy_utility_tradeoff(results_dict, save_path=None):
    """绘制隐私-效用曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    markers = {"uniform": "o", "decreasing": "s", "adaptive": "^",
               "LightGCN(NoDP)": "D"}
    colors = {"uniform": "#E74C3C", "decreasing": "#3498DB", "adaptive": "#2ECC71",
              "LightGCN(NoDP)": "#555555"}

    for strategy, results in results_dict.items():
        epsilons = sorted(results.keys())
        recalls = [results[eps].get("Recall@20", 0) for eps in epsilons]
        ndcgs = [results[eps].get("NDCG@20", 0) for eps in epsilons]

        m = markers.get(strategy, "o")
        c = colors.get(strategy, "#333333")
        label = strategy.replace("LightGCN(NoDP)", "LightGCN (No DP)")
        axes[0].plot(epsilons, recalls, marker=m, color=c, label=label, linewidth=2, markersize=6)
        axes[1].plot(epsilons, ndcgs, marker=m, color=c, label=label, linewidth=2, markersize=6)

    axes[0].set_xlabel("Epsilon (privacy budget)")
    axes[0].set_ylabel("Recall@20")
    axes[0].set_title("Privacy-Utility Trade-off (Recall@20)")
    axes[0].legend(fontsize=9)

    axes[1].set_xlabel("Epsilon (privacy budget)")
    axes[1].set_ylabel("NDCG@20")
    axes[1].set_title("Privacy-Utility Trade-off (NDCG@20)")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.savefig(save_path.replace(".pdf", ".png"), format="png", bbox_inches="tight", dpi=200)
        print(f"[Figure] Saved to {save_path}")
    plt.close()


def plot_comparison_bar(results_df, save_path=None):
    """绘制不同方法的对比柱状图"""
    epsilons = sorted(results_df["Epsilon"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, metric in enumerate(["Recall@20", "NDCG@20"]):
        ax = axes[idx]
        bar_width = 0.12
        models = results_df["Model"].unique()
        x = np.arange(len(epsilons))

        for i, model in enumerate(models):
            subset = results_df[results_df["Model"] == model]
            values = [subset[subset["Epsilon"] == e][metric].values[0] if e in subset["Epsilon"].values else 0 for e in epsilons]
            offset = (i - len(models) / 2 + 0.5) * bar_width
            ax.bar(x + offset, values, bar_width, label=model)

        ax.set_xlabel("Epsilon")
        ax.set_ylabel(metric)
        ax.set_title(f"Comparison: {metric}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(e) for e in epsilons])
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"[Figure] Saved to {save_path}")
    plt.close()


def save_results_csv(all_results, save_path="./results/experiment_results.csv"):
    """保存实验结果到CSV"""
    rows = []
    for strategy, eps_results in all_results.items():
        for eps, metrics in eps_results.items():
            eps_display = float("inf") if eps >= 1e6 else eps
            rows.append({
                "Model": strategy,
                "Epsilon": eps_display,
                "Recall@20": metrics.get("Recall@20", 0),
                "NDCG@20": metrics.get("NDCG@20", 0),
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[Results] Saved to {save_path}")
    return df
