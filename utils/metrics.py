"""
评估指标：Recall@K, NDCG@K
"""
import numpy as np
import torch

def get_metrics(model, dataset, sampler, top_k_list=[20], device="cpu"):
    model.eval()
    with torch.no_grad():
        all_embeddings = model(dataset.get_adj_tensor(device))
        user_embeddings = all_embeddings[:dataset.n_users]
        item_embeddings = all_embeddings[dataset.n_users:]
        recall_list = {k: [] for k in top_k_list}
        ndcg_list = {k: [] for k in top_k_list}
        for user, pos_items in dataset.test_data:
            user_tensor = torch.LongTensor([user]).to(device)
            u_emb = user_embeddings[user_tensor]
            scores = torch.matmul(u_emb, item_embeddings.t()).squeeze()
            train_pos = list(sampler.user_pos_dict[user])
            scores[train_pos] = -1e9
            _, indices = torch.topk(scores, max(top_k_list))
            indices = indices.cpu().numpy()
            test_item = pos_items[-1]
            for k in top_k_list:
                top_k_items = indices[:k]
                if test_item in top_k_items:
                    recall_list[k].append(1)
                    rank = np.where(top_k_items == test_item)[0][0] + 1
                    ndcg_list[k].append(1.0 / np.log2(rank + 1))
                else:
                    recall_list[k].append(0)
                    ndcg_list[k].append(0)
    results = {}
    for k in top_k_list:
        rec_key = "Recall@{}".format(k)
        ndcg_key = "NDCG@{}".format(k)
        results[rec_key] = np.mean(recall_list[k]) if recall_list[k] else 0.0
        results[ndcg_key] = np.mean(ndcg_list[k]) if ndcg_list[k] else 0.0
    return results

def print_metrics(results):
    for k in sorted([int(key.split("@")[1]) for key in results.keys() if "Recall" in key]):
        rec = results.get("Recall@{}".format(k), 0.0)
        ndcg = results.get("NDCG@{}".format(k), 0.0)
        print("  Recall@{:2d} = {:.6f}  NDCG@{:2d} = {:.6f}".format(k, rec, k, ndcg))
