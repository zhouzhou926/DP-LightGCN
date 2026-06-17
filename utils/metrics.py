"""
评估指标：Recall@K, NDCG@K（GPU分批计算→CPU处理，避免显存溢出）
"""
import numpy as np
import torch

def get_metrics(model, dataset, sampler, top_k_list=[20], device="cpu"):
    model.eval()
    with torch.no_grad():
        all_embeddings = model(dataset.get_adj_tensor(device))
        u_emb = all_embeddings[:dataset.n_users]
        i_emb = all_embeddings[dataset.n_users:]
        
        recall_list = {k: [] for k in top_k_list}
        ndcg_list = {k: [] for k in top_k_list}
        
        # 分批在GPU上计算评分，然后转移到CPU处理
        batch_size = 1000
        n_users = dataset.n_users
        
        # 先构建训练集mask索引（避免每次在GPU上操作）
        user_masks = {}
        for user, items in dataset.test_data:
            train_items = sampler.user_pos_dict.get(user, set())
            test_set = set(items)
            mask = [i for i in train_items if i not in test_set]
            if mask:
                user_masks[user] = mask
        
        for start in range(0, n_users, batch_size):
            end = min(start + batch_size, n_users)
            scores_gpu = torch.matmul(u_emb[start:end], i_emb.t())  # [batch, 40981]
            scores_np = scores_gpu.cpu().numpy()  # 转移到CPU
            del scores_gpu  # 释放GPU显存
            
            for local_idx in range(end - start):
                user = start + local_idx
                test_set, n_test = dataset.test_data_idx.get(user, (None, 0))
                if test_set is None or n_test == 0:
                    continue
                
                scores = scores_np[local_idx].copy()
                mask = user_masks.get(user)
                if mask:
                    scores[mask] = -1e9
                
                max_k = max(top_k_list)
                topk_idx = np.argsort(scores)[-max_k:][::-1]
                
                for k in top_k_list:
                    top_k_set = set(topk_idx[:k])
                    hits = top_k_set & test_set
                    n_hits = len(hits)
                    recall_list[k].append(n_hits / n_test)
                    
                    dcg = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(topk_idx[:k]) if item in test_set)
                    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(n_test, k)))
                    ndcg_list[k].append(dcg / idcg if idcg > 0 else 0.0)
        
        results = {}
        for k in top_k_list:
            results[f"Recall@{k}"] = float(np.mean(recall_list[k])) if recall_list[k] else 0.0
            results[f"NDCG@{k}"] = float(np.mean(ndcg_list[k])) if ndcg_list[k] else 0.0
        return results


def print_metrics(results):
    for k in sorted([int(key.split("@")[1]) for key in results.keys() if "Recall" in key]):
        rec = results.get(f"Recall@{k}", 0.0)
        ndcg = results.get(f"NDCG@{k}", 0.0)
        print(f"  Recall@{k:2d} = {rec:.6f}  NDCG@{k:2d} = {ndcg:.6f}")
