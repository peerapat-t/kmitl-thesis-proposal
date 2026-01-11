import pandas as pd
import numpy as np
from typing import Tuple


def get_loo_split(
    df: pd.DataFrame,
    sparse_threshold: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Step 0) Validate inputs
    required_cols = {"userid", "itemid", "timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    sparse_threshold = float(sparse_threshold)
    if not (0.0 <= sparse_threshold <= 1.0):
        raise ValueError("sparse_threshold must be in [0, 1] (e.g., 0.2 for bottom 20%)")

    # Step 1) Copy + ensure rating exists
    df = df.copy()
    if "rating" not in df.columns:
        df["rating"] = 1.0

    # Step 2) Sort per user by time (old -> new)
    df = df.sort_values(["userid", "timestamp"]).reset_index(drop=True)

    # Step 3) Map raw ids -> contiguous indices
    df["u_idx"] = df["userid"].astype("category").cat.codes
    df["i_idx"] = df["itemid"].astype("category").cat.codes

    n_users = int(df["u_idx"].max()) + 1
    n_items = int(df["i_idx"].max()) + 1

    # Step 4) Rank by recency (1 = latest)
    df["rank_latest"] = df.groupby("u_idx")["timestamp"].rank(method="first", ascending=False)

    # Step 5) LOO split frames
    test_df = df[df["rank_latest"] == 1]
    val_df = df[df["rank_latest"] == 2]
    train_df = df[df["rank_latest"] > 2]

    # Step 6) Init matrices
    train_mat = np.zeros((n_users, n_items), dtype=np.float32)
    val_mat = np.zeros((n_users, n_items), dtype=np.float32)
    test_mat = np.zeros((n_users, n_items), dtype=np.float32)

    # Step 7) Fill matrices (max for duplicates)
    def fill_matrix_max(mat: np.ndarray, data: pd.DataFrame) -> None:
        if data.empty:
            return
        u = data["u_idx"].to_numpy()
        i = data["i_idx"].to_numpy()
        r = data["rating"].to_numpy(dtype=np.float32)
        np.maximum.at(mat, (u, i), r)

    fill_matrix_max(train_mat, train_df)
    fill_matrix_max(val_mat, val_df)
    fill_matrix_max(test_mat, test_df)

    train_val_mat = np.maximum(train_mat, val_mat)

    # Step 8) Bottom-fraction indices (strict k, stable order)
    def bottom_fraction_indices(counts: np.ndarray, frac: float) -> np.ndarray:
        if frac <= 0.0:
            return np.array([], dtype=np.int64)
        if frac >= 1.0:
            return np.arange(counts.shape[0], dtype=np.int64)
        k = int(np.ceil(frac * counts.shape[0]))
        order = np.argsort(counts, kind="stable")
        return order[:k].astype(np.int64)

    train_counts = np.count_nonzero(train_mat, axis=1)
    sparse_train_users = bottom_fraction_indices(train_counts, sparse_threshold)

    train_val_counts = np.count_nonzero(train_val_mat, axis=1)
    sparse_train_val_users = bottom_fraction_indices(train_val_counts, sparse_threshold)

    # Step 9) Return
    return (
        train_mat,
        val_mat,
        sparse_train_users,
        train_val_mat,
        test_mat,
        sparse_train_val_users,
    )