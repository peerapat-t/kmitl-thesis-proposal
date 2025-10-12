import numpy as np
from typing import List, Set, Tuple

def ap_at_k(predicted: List[int], actual: Set[int], k: int) -> float:
    if not actual:
        return 0.0

    score = 0.0
    num_hits = 0.0
    
    for i, p in enumerate(predicted[:k]):
        if p in actual:
            num_hits += 1.0
            precision_at_i = num_hits / (i + 1.0)
            score += precision_at_i

    return score / min(len(actual), k)


def map_at_k(
    pred_array: np.ndarray, 
    test_array: np.ndarray, 
    cold_start_flag_array: np.ndarray, 
    k: int
) -> Tuple[float, float, float]:
    
    ap_scores = []
    ap_scores_cold = []
    ap_scores_not_cold = []
    
    num_users = pred_array.shape[0]

    for u in range(num_users):
        actual_indices = set(np.where(test_array[u, :] > 0)[0])

        if len(actual_indices) == 0:
            continue

        pred_scores_u = pred_array[u, :]
        top_k_indices = np.argsort(-pred_scores_u)[:k]

        ap = ap_at_k(list(top_k_indices), actual_indices, k)

        ap_scores.append(ap)
        if cold_start_flag_array[u]:
            ap_scores_cold.append(ap)
        else:
            ap_scores_not_cold.append(ap)

    map_k = np.mean(ap_scores) if ap_scores else 0.0
    map_k_cold = np.mean(ap_scores_cold) if ap_scores_cold else 0.0
    map_k_not_cold = np.mean(ap_scores_not_cold) if ap_scores_not_cold else 0.0

    return map_k, map_k_cold, map_k_not_cold