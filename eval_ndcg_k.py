import numpy as np
from typing import List, Set, Tuple

def dcg_at_k(predicted: List[int], actual: Set[int], k: int) -> float:
    dcg = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual:
            dcg += 1.0 / np.log2(i + 2)
    return dcg

def ndcg_at_k(predicted: List[int], actual: Set[int], k: int) -> float:
    if not actual:
        return 0.0
    
    dcg = dcg_at_k(predicted, actual, k)
    
    num_possible_hits = min(len(actual), k)
    idcg = 0.0
    for i in range(num_possible_hits):
        idcg += 1.0 / np.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg

def calculate_ndcg_at_k(
    pred_array: np.ndarray, 
    train_array: np.ndarray,
    test_array: np.ndarray, 
    cold_start_flag_array: np.ndarray, 
    k: int
) -> Tuple[float, float, float]:
    
    ndcg_scores = []
    ndcg_scores_cold = []
    ndcg_scores_not_cold = []
    
    num_users = pred_array.shape[0]

    for u in range(num_users):
        actual_indices = set(np.where(test_array[u, :] > 0)[0])

        if not actual_indices:
            continue

        pred_scores_u = pred_array[u, :].copy()
        
        train_mask_u = (train_array[u, :] > 0)
        pred_scores_u[train_mask_u] = -np.inf
        
        top_k_indices = np.argsort(-pred_scores_u)[:k]
        
        ndcg_val = ndcg_at_k(list(top_k_indices), actual_indices, k)

        ndcg_scores.append(ndcg_val)
        
        if cold_start_flag_array[u]:
            ndcg_scores_cold.append(ndcg_val)
        else:
            ndcg_scores_not_cold.append(ndcg_val)
    
    mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    mean_ndcg_cold = np.mean(ndcg_scores_cold) if ndcg_scores_cold else 0.0
    mean_ndcg_not_cold = np.mean(ndcg_scores_not_cold) if ndcg_scores_not_cold else 0.0

    return mean_ndcg, mean_ndcg_cold, mean_ndcg_not_cold