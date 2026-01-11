import numpy as np


def ndcg_at_k(
    pred_mat: np.ndarray,
    train_mat: np.ndarray,
    test_mat: np.ndarray,
    cold_user_indices=None,
    ks=(5, 10, 20),
    mask_train: bool = True,
):
    # Step 1) Convert inputs to numpy arrays + validate shapes
    pred_mat = np.asarray(pred_mat, dtype=float)
    train_mat = np.asarray(train_mat)
    test_mat = np.asarray(test_mat)

    U, I = pred_mat.shape
    if train_mat.shape != (U, I) or test_mat.shape != (U, I):
        raise ValueError("pred_mat, train_mat, test_mat must have the same shape (U, I)")

    # Step 2) Normalize ks and precompute discounts + IDCG cumulative
    ks = tuple(int(k) for k in ks)
    if len(ks) == 0 or min(ks) <= 0:
        raise ValueError("ks must contain positive integers")

    ks = tuple(min(k, I) for k in ks)
    k_max = min(max(ks), I)

    discounts = 1.0 / np.log2(np.arange(2, k_max + 2, dtype=float))
    idcg_cum = np.zeros(k_max + 1, dtype=float)
    idcg_cum[1:] = np.cumsum(discounts)

    # Step 3) Build cold user mask
    cold_mask = np.zeros(U, dtype=bool)
    if cold_user_indices is not None:
        cold_user_indices = np.asarray(list(cold_user_indices), dtype=int)
        cold_user_indices = cold_user_indices[(cold_user_indices >= 0) & (cold_user_indices < U)]
        cold_mask[cold_user_indices] = True

    # Step 4) Short-circuit if masking train but pred_mat == train_mat (all masked)
    if mask_train and np.array_equal(pred_mat, train_mat):
        has_test = (test_mat > 0).any(axis=1)
        cnt_all = int(has_test.sum())
        cnt_cold = int((has_test & cold_mask).sum())
        cnt_warm = int((has_test & (~cold_mask)).sum())

        zero = {k: 0.0 for k in ks}
        return {
            "all": dict(zero),
            "cold": dict(zero),
            "warm": dict(zero),
            "n_users_eval": cnt_all,
            "n_cold_eval": cnt_cold,
            "n_warm_eval": cnt_warm,
            "ks": list(ks),
        }

    # Step 5) Initialize accumulators
    sums_all = np.zeros(len(ks), dtype=float)
    sums_cold = np.zeros(len(ks), dtype=float)
    sums_warm = np.zeros(len(ks), dtype=float)
    cnt_all = 0
    cnt_cold = 0
    cnt_warm = 0

    # Step 6) Evaluate per user
    for u in range(U):
        targets = np.where(test_mat[u] > 0)[0]
        if targets.size == 0:
            continue

        # Step 6.1) Copy scores and mask seen training items (optional)
        scores = pred_mat[u].copy()
        if mask_train:
            scores[train_mat[u] > 0] = -np.inf
        if not np.isfinite(scores).any():
            continue

        # Step 6.2) Extract top-k_max indices by score (descending)
        topk = np.argpartition(scores, -k_max)[-k_max:]
        topk = topk[np.argsort(scores[topk])[::-1]]

        # Step 6.3) Compute DCG cumulative using hits and discounts
        hits = np.isin(topk, targets, assume_unique=False)
        dcg_cum = np.cumsum(hits.astype(float) * discounts)

        # Step 6.4) Compute NDCG@k for each k in ks
        ndcgs = np.empty(len(ks), dtype=float)
        for j, k in enumerate(ks):
            dcg = float(dcg_cum[k - 1])
            m = k if targets.size >= k else int(targets.size)
            idcg = float(idcg_cum[m])
            ndcgs[j] = (dcg / idcg) if idcg > 0 else 0.0

        # Step 6.5) Accumulate (all / cold / warm)
        sums_all += ndcgs
        cnt_all += 1

        if cold_mask[u]:
            sums_cold += ndcgs
            cnt_cold += 1
        else:
            sums_warm += ndcgs
            cnt_warm += 1

    # Step 7) Convert sums to mean dicts
    def to_dict(sums, cnt):
        if cnt == 0:
            return {k: 0.0 for k in ks}
        mean_vals = sums / cnt
        return {k: float(mean_vals[i]) for i, k in enumerate(ks)}

    # Step 8) Return summary
    return {
        "all": to_dict(sums_all, cnt_all),
        "cold": to_dict(sums_cold, cnt_cold),
        "warm": to_dict(sums_warm, cnt_warm),
        "n_users_eval": int(cnt_all),
        "n_cold_eval": int(cnt_cold),
        "n_warm_eval": int(cnt_warm),
        "ks": list(ks),
    }


def hit_rate_at_k(
    pred_mat: np.ndarray,
    train_mat: np.ndarray,
    test_mat: np.ndarray,
    cold_user_indices=None,
    ks=(5, 10, 20),
    mask_train: bool = True,
):
    # Step 1) Convert inputs to numpy arrays + validate shapes
    pred_mat = np.asarray(pred_mat, dtype=float)
    train_mat = np.asarray(train_mat)
    test_mat = np.asarray(test_mat)

    U, I = pred_mat.shape
    if train_mat.shape != (U, I) or test_mat.shape != (U, I):
        raise ValueError("pred_mat, train_mat, test_mat must have the same shape (U, I)")

    # Step 2) Normalize ks and compute k_max
    ks = tuple(int(k) for k in ks)
    if len(ks) == 0 or min(ks) <= 0:
        raise ValueError("ks must contain positive integers")

    ks = tuple(min(k, I) for k in ks)
    k_max = min(max(ks), I)

    # Step 3) Build cold user mask
    cold_mask = np.zeros(U, dtype=bool)
    if cold_user_indices is not None:
        cold_user_indices = np.asarray(list(cold_user_indices), dtype=int)
        cold_user_indices = cold_user_indices[(cold_user_indices >= 0) & (cold_user_indices < U)]
        cold_mask[cold_user_indices] = True

    # Step 4) Short-circuit if masking train but pred_mat == train_mat (all masked)
    if mask_train and np.array_equal(pred_mat, train_mat):
        has_test = (test_mat > 0).any(axis=1)
        cnt_all = int(has_test.sum())
        cnt_cold = int((has_test & cold_mask).sum())
        cnt_warm = int((has_test & (~cold_mask)).sum())

        zero = {k: 0.0 for k in ks}
        return {
            "all": dict(zero),
            "cold": dict(zero),
            "warm": dict(zero),
            "n_users_eval": cnt_all,
            "n_cold_eval": cnt_cold,
            "n_warm_eval": cnt_warm,
            "ks": list(ks),
        }

    # Step 5) Initialize accumulators
    sums_all = np.zeros(len(ks), dtype=float)
    sums_cold = np.zeros(len(ks), dtype=float)
    sums_warm = np.zeros(len(ks), dtype=float)
    cnt_all = 0
    cnt_cold = 0
    cnt_warm = 0

    # Step 6) Evaluate per user
    for u in range(U):
        targets = np.where(test_mat[u] > 0)[0]
        if targets.size == 0:
            continue

        # Step 6.1) Copy scores and mask seen training items (optional)
        scores = pred_mat[u].copy()
        if mask_train:
            scores[train_mat[u] > 0] = -np.inf

        # Step 6.2) Determine finite candidates and k_u
        finite_mask = np.isfinite(scores)
        n_finite = int(finite_mask.sum())
        if n_finite == 0:
            continue
        k_u = min(k_max, n_finite)

        # Step 6.3) Extract top-k_u indices by score (descending; deterministic tie-break by index)
        topk = np.argpartition(scores, -k_u)[-k_u:]
        order = np.lexsort((topk, -scores[topk]))
        topk = topk[order]

        # Step 6.4) Compute cumulative "hit any" vector
        hits = np.isin(topk, targets, assume_unique=False)
        hit_any_cum = np.maximum.accumulate(hits)

        # Step 6.5) Compute HR@k for each k in ks
        hrs = np.empty(len(ks), dtype=float)
        for j, k in enumerate(ks):
            kk = min(k, topk.size)
            if kk <= 0:
                hrs[j] = 0.0
            else:
                hrs[j] = float(hit_any_cum[kk - 1])

        # Step 6.6) Accumulate (all / cold / warm)
        sums_all += hrs
        cnt_all += 1

        if cold_mask[u]:
            sums_cold += hrs
            cnt_cold += 1
        else:
            sums_warm += hrs
            cnt_warm += 1

    # Step 7) Convert sums to mean dicts
    def to_dict(sums, cnt):
        if cnt == 0:
            return {k: 0.0 for k in ks}
        mean_vals = sums / cnt
        return {k: float(mean_vals[i]) for i, k in enumerate(ks)}

    # Step 8) Return summary
    return {
        "all": to_dict(sums_all, cnt_all),
        "cold": to_dict(sums_cold, cnt_cold),
        "warm": to_dict(sums_warm, cnt_warm),
        "n_users_eval": int(cnt_all),
        "n_cold_eval": int(cnt_cold),
        "n_warm_eval": int(cnt_warm),
        "ks": list(ks),
    }