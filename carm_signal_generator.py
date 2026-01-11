import numpy as np
from sklearn.cluster import KMeans


class SignalGeneratorClusterARM_v15:
    def __init__(
        self,
        k_user=10,
        min_support=0.002,
        min_confidence=0.05,
        remove_seen=True,
        random_state=42,
        lift_clip_max=10.0,
        eps=1e-12,
    ):
        # Step 0) Store hyperparameters + constants
        self.k_user = int(k_user)
        self.min_support = float(min_support)
        self.min_confidence = float(min_confidence)
        self.remove_seen = bool(remove_seen)
        self.random_state = int(random_state)
        self.eps = float(eps)
        self.lift_clip_max = float(lift_clip_max)

        # Step 0.1) Learned artifacts
        self.user_labels_ = None
        self.user_medians_ = None
        self.cluster_non_all_zero_counts_ = None
        self.cluster_all_counts_ = None
        self.cluster_non_cold_counts_ = None
        self.active_items_ = None
        self.k_ = None
        self.score_ante_to_item_ = None

    @staticmethod
    def _ternary_from_Y_and_rated(Y: np.ndarray, rated: np.ndarray):
        # Step A1) Identify users with any observed ratings
        U = Y.shape[0]
        has_rated = rated.any(axis=1)

        # Step A2) Compute per-user median over observed entries only
        med = np.zeros(U, dtype=float)
        if np.any(has_rated):
            Y_nan = np.where(rated, Y, np.nan)
            med[has_rated] = np.nanmedian(Y_nan[has_rated], axis=1)

        # Step A3) Create like/dislike masks using the medians
        like = rated & (Y >= med[:, None])
        dislike = rated & (Y < med[:, None])

        # Step A4) Build ternary matrix Z
        Z = np.zeros_like(Y, dtype=np.int8)
        Z[like] = 1
        Z[dislike] = -1
        return Z, med

    def fit_transform(self, Y):
        # Step 1) Validate input
        Y = np.asarray(Y, dtype=float)
        if (Y < 0).any() or (~np.isfinite(Y)).any():
            raise ValueError("Y must be finite and >= 0")

        # Step 2) Read matrix shape
        U, I = Y.shape
        if U == 0 or I == 0:
            return np.zeros((U, I), dtype=float)

        # Step 3) Compute rated mask ONCE
        rated = (Y > 0)
        O = rated.astype(np.int8)

        # Step 4) Compute per-user observation counts
        obs_cnt = O.sum(axis=1)

        # Step 5) Build ternary matrix Z and per-user medians
        Z, med = self._ternary_from_Y_and_rated(Y, rated)
        self.user_medians_ = med

        # Step 6) Split users into non-all-zero vs all-zero
        all_zero_mask = (obs_cnt == 0)
        non_all_zero_idx = np.where(~all_zero_mask)[0]
        all_zero_idx = np.where(all_zero_mask)[0]

        # Step 7) Handle degenerate case: no non-all-zero users
        if non_all_zero_idx.size == 0:
            S = np.zeros((U, I), dtype=float)
            if self.remove_seen:
                S[rated] = 0.0
            self.user_labels_ = np.zeros(U, dtype=int)
            self.cluster_non_all_zero_counts_ = np.array([0], dtype=int)
            self.cluster_all_counts_ = np.array([U], dtype=int)
            self.cluster_non_cold_counts_ = self.cluster_non_all_zero_counts_
            self.active_items_ = np.array([], dtype=int)
            self.k_ = 1
            self.score_ante_to_item_ = None
            return S

        # Step 8) KMeans clustering on non-all-zero users
        k = min(self.k_user, non_all_zero_idx.size)
        self.k_ = k
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)

        # Step 9) Select features for clustering (avoid all-zero columns)
        Z_fit = Z[non_all_zero_idx]
        feat_keep = np.any(Z_fit != 0, axis=0)
        if not np.any(feat_keep):
            feat_keep = np.zeros(I, dtype=bool)
            feat_keep[0] = True

        # Step 10) Fit and assign cluster labels
        Z_fit2 = Z_fit[:, feat_keep].astype(float, copy=False)
        labels_fit = km.fit_predict(Z_fit2)
        labels_all = np.empty(U, dtype=int)
        labels_all[non_all_zero_idx] = labels_fit

        # Step 11) Assign all-zero users to nearest centroid of the zero-vector
        if all_zero_idx.size > 0:
            z0 = np.zeros((1, int(feat_keep.sum())), dtype=float)
            label_all_zero = int(km.predict(z0)[0])
            labels_all[all_zero_idx] = label_all_zero

        # Step 12) Save labels and cluster counts
        self.user_labels_ = labels_all
        self.cluster_all_counts_ = np.bincount(labels_all, minlength=k).astype(int)
        self.cluster_non_all_zero_counts_ = np.bincount(
            labels_all[non_all_zero_idx], minlength=k
        ).astype(int)
        self.cluster_non_cold_counts_ = self.cluster_non_all_zero_counts_

        # Step 13) Prepare ARM inputs using non-all-zero users only
        O_warm = O[non_all_zero_idx].astype(np.float32, copy=False)
        labels_warm = labels_all[non_all_zero_idx]
        Nf = float(O_warm.shape[0])

        # Step 14) Determine active items
        pop_item = O_warm.sum(axis=0)
        active_items = np.flatnonzero(pop_item > 0)
        self.active_items_ = active_items

        # Step 15) Initialize score matrix S
        S = np.zeros((U, I), dtype=np.float32)

        # Step 16) Handle degenerate case: no active items
        if active_items.size == 0:
            if self.remove_seen:
                S = S.copy()
                S[rated] = 0.0
            self.score_ante_to_item_ = None
            return S.astype(float, copy=False)

        # Step 17) Build X = [Oa | cluster-onehot]
        Oa = O_warm[:, active_items]
        Ia = int(Oa.shape[1])
        X = np.zeros((Oa.shape[0], Ia + k), dtype=np.float32)
        X[:, :Ia] = Oa
        rows = np.arange(Oa.shape[0], dtype=np.int64)
        X[rows, Ia + labels_warm] = 1.0

        # Step 18) Compute co-occurrence
        co_full = (X.T @ X).astype(np.float32, copy=False)
        co_all = co_full[:, :Ia]
        pop_all = X.sum(axis=0).astype(np.float32, copy=False)
        pop_all_safe = np.maximum(pop_all, 1.0)

        # Step 19) Compute confidence and support
        conf_all = co_all / pop_all_safe[:, None]
        sup_all = co_all / max(Nf, 1.0)

        # Step 20) Compute sqrt-lift with clipping
        p_item = (pop_all[:Ia] / max(Nf, 1.0)).astype(np.float32, copy=False)
        lift_raw = conf_all / np.maximum(p_item[None, :], float(self.eps))
        lift_raw = np.clip(lift_raw, 0.0, self.lift_clip_max)
        lift_all = np.sqrt(lift_raw)

        # Step 21) Apply threshold mask and remove item->same item rules
        mask = (sup_all >= self.min_support) & (conf_all >= self.min_confidence)
        diag = np.arange(Ia, dtype=np.int64)
        mask[diag, diag] = False

        # Step 22) Build rule weight matrix W
        W = (conf_all * lift_all) * mask
        self.score_ante_to_item_ = W

        # Step 23) Score warm users
        S_warm_a = X @ W
        S[np.ix_(non_all_zero_idx, active_items)] = S_warm_a

        # Step 24) Score all-zero users using cluster->item row only
        if all_zero_idx.size > 0:
            cl0 = labels_all[all_zero_idx]
            S[np.ix_(all_zero_idx, active_items)] = W[Ia + cl0, :]

        # Step 25) Remove seen items
        if self.remove_seen:
            S = S.copy()
            S[rated] = 0.0

        # Step 26) Return scores
        return S.astype(float, copy=False)