
import sys; sys.path.insert(0, ".")
import torch, numpy as np
from models.dp_mechanism import EmbeddingPerturbation

pert = EmbeddingPerturbation(10.0, 1e-5, 1.0, "uniform")
for strategy in ["uniform", "decreasing", "adaptive"]:
    pert.budget_strategy = strategy
    eps = pert.compute_layer_budgets(3)
    sigmas = [pert.get_noise_scale(e) for e in eps]
    print(f"{strategy:10s} | eps: {[f'{e:.2f}' for e in eps]} | sigma: {[f'{s:.4f}' for s in sigmas]}")

# 嵌入范数估计
print()
print(f"Embedding norm (init, ~sqrt(64)*0.1) = {np.sqrt(64)*0.1:.4f}")
sigma_10 = pert.get_noise_scale(10.0/3)
print(f"Noise sigma (uniform, eps=3.33): {sigma_10:.4f}")
print(f"SNR: {np.sqrt(64)*0.1/sigma_10:.2f}x")
print(f"噪声强度是信号的 {100*sigma_10/(np.sqrt(64)*0.1):.0f}%")
