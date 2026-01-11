import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Step 1) Dataset: each batch returns (user_idx, Y[user_idx], S[user_idx])
class PrepareDataset_StandardV2(Dataset):
    def __init__(self, Y_tensor: torch.Tensor, S_tensor: torch.Tensor):
        self.Y = Y_tensor
        self.S = S_tensor

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, user_idx: int):
        return user_idx, self.Y[user_idx], self.S[user_idx]


# Step 2) CARMS MF
class CARMS_Bias_MF(nn.Module):
    def __init__(self, user_count, item_count, K, learning_rate, lambda_rate, gamma, seed=42):
        super().__init__()
        torch.manual_seed(seed)

        self.user_count = int(user_count)
        self.item_count = int(item_count)
        self.K = int(K)
        self.learning_rate = float(learning_rate)
        self.lambda_rate = float(lambda_rate)
        self.gamma = float(gamma)

        # Step 2.1) Learnable parameters
        self.P = nn.Embedding(self.user_count, self.K)
        self.Q = nn.Embedding(self.item_count, self.K)
        self.b_u = nn.Embedding(self.user_count, 1)
        self.b_i = nn.Embedding(self.item_count, 1)
        self.mu = nn.Parameter(torch.zeros(1))

        # Step 2.2) Initialization
        nn.init.normal_(self.P.weight, std=0.01)
        nn.init.normal_(self.Q.weight, std=0.01)
        nn.init.zeros_(self.b_u.weight)
        nn.init.zeros_(self.b_i.weight)

        # Step 2.3) Device selection
        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

    # Step 3) Forward calculation: produce scores for all items for a batch of users
    def forward(self, user_idx: torch.LongTensor) -> torch.Tensor:
        # Step 3.1) Gather user factors and user biases for the batch
        Pu = self.P(user_idx)
        bu = self.b_u(user_idx)

        # Step 3.2) Use all item factors and item biases
        Qt = self.Q.weight
        bi = self.b_i.weight.view(1, -1)

        # Step 3.3) Compute full score matrix for the batch: (B, I)
        return self.mu + bu + bi + (Pu @ Qt.T)

    # Step 4) Confidence transform: sanitize and clamp to non-negative
    @staticmethod
    def _transform_confidence(S: torch.Tensor) -> torch.Tensor:
        # Step 4.1) Replace nan/inf with 0
        S = torch.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        # Step 4.2) Enforce S >= 0
        return torch.clamp(S, min=0.0)

    # Step 5) Sample one positive index per valid user-row from weights W (B, I)
    @staticmethod
    def _sample_pos(W: torch.Tensor):
        # Step 5.1) Identify rows that have at least one positive weight
        row_sum = W.sum(dim=1)
        valid = row_sum > 0
        if not valid.any():
            return valid, None
        # Step 5.2) Multinomial sampling within each valid row
        pos_idx = torch.multinomial(W[valid], num_samples=1).squeeze(1)
        return valid, pos_idx

    # Step 6) Sample one negative index per row from boolean mask neg_mask (B, I)
    @staticmethod
    def _sample_neg(neg_mask: torch.Tensor, device: torch.device):
        # Step 6.1) Convert mask -> weights and check which rows have available negatives
        neg_w = neg_mask.float()
        row_sum = neg_w.sum(dim=1)
        has_neg = row_sum > 0

        # Step 6.2) Allocate output indices
        neg_idx = torch.empty((neg_mask.size(0),), dtype=torch.long, device=device)

        # Step 6.3) If negatives exist: sample from them
        if has_neg.any():
            neg_idx[has_neg] = torch.multinomial(neg_w[has_neg], num_samples=1).squeeze(1)

        # Step 6.4) Fallback: if a row has no negatives, sample uniformly at random
        if (~has_neg).any():
            neg_idx[~has_neg] = torch.randint(
                0, neg_mask.size(1), (int((~has_neg).sum().item()),), device=device
            )

        return neg_idx, has_neg

    # Step 7) Training loop: SSE on observed Y + gamma * BPR(sum) on confidence positives
    def fit(self, Y: np.ndarray, S: np.ndarray, epochs: int, batch_size: int = 512, verbose_every: int = 10):
        # Step 7.1) Validate input arrays
        Y = np.asarray(Y, dtype=np.float32)
        S = np.asarray(S, dtype=np.float32)

        if Y.shape != (self.user_count, self.item_count) or S.shape != (self.user_count, self.item_count):
            raise ValueError(f"Y,S must be shape {(self.user_count, self.item_count)}")
        if not np.isfinite(Y).all() or not np.isfinite(S).all():
            raise ValueError("Y or S has NaN/Inf.")
        if Y.min() < 0:
            raise ValueError("Y has negative values.")

        # Step 7.2) Move model to device and set train mode
        self.to(self.device)
        self.train()

        # Step 7.3) Build tensors (kept on CPU; batches moved to device during training)
        Y_tensor = torch.from_numpy(Y)
        S_tensor = torch.from_numpy(S)

        # Step 7.4) Initialize mu using observed mean
        if (Y_tensor != 0).any():
            with torch.no_grad():
                self.mu.fill_(float(Y_tensor[Y_tensor != 0].mean().item()))

        # Step 7.5) DataLoader over users
        loader = DataLoader(
            PrepareDataset_StandardV2(Y_tensor, S_tensor),
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=0,
        )

        # Step 7.6) Optimizer (weight_decay as L2 on embeddings only)
        optimizer = torch.optim.Adam(
            [
                {"params": self.P.weight, "weight_decay": self.lambda_rate},
                {"params": self.Q.weight, "weight_decay": self.lambda_rate},
                {"params": self.b_u.weight, "weight_decay": 0.0},
                {"params": self.b_i.weight, "weight_decay": 0.0},
                {"params": self.mu, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
        )

        # Step 8) Epoch loop
        for epoch in range(int(epochs)):
            total_sse_obs = 0.0
            total_bpr_sum = 0.0
            num_obs = 0
            num_pairs = 0

            # Step 8.1) Batch loop
            for user_idx, Yb, Sb in loader:
                # Step 8.2) Move batch to device
                user_idx = user_idx.to(self.device)
                Yb = Yb.to(self.device)
                Sb = Sb.to(self.device)

                # Step 8.3) Predict all items for these users
                pred = self(user_idx)  # (B, I)

                # Step 8.4) SSE on observed entries (Y != 0)
                mask_obs = (Yb != 0)
                has_obs = mask_obs.any()
                if has_obs:
                    err = Yb[mask_obs] - pred[mask_obs]
                    sse_obs = torch.sum(err * err)
                    total_sse_obs += float(sse_obs.item())
                    num_obs += int(mask_obs.sum().item())
                else:
                    sse_obs = torch.tensor(0.0, device=self.device)

                # Step 8.5) Confidence transform S -> Sconf >= 0
                Sconf = self._transform_confidence(Sb)

                # Step 8.6) Define positive candidates for BPR: (Y==0) AND (Sconf>0)
                pos_mask = (Yb == 0) & (Sconf > 0)

                # Step 8.7) Sample one positive per user-row (weighted by pos_mask)
                valid, pos_idx = self._sample_pos(pos_mask.float())

                # Step 8.8) If we have valid rows: sample negatives and compute BPR(sum)
                if valid.any():
                    pred_v = pred[valid]     # (Bv, I)
                    Y_v = Yb[valid]          # (Bv, I)
                    S_v = Sconf[valid]       # (Bv, I)

                    # Step 8.8.1) Negatives: items with (Y==0) AND (Sconf==0)
                    neg_mask = (Y_v == 0) & (S_v == 0)
                    neg_idx, _ = self._sample_neg(neg_mask, device=self.device)

                    # Step 8.8.2) Gather pos/neg predictions per row
                    u_local = torch.arange(pred_v.size(0), device=self.device)
                    pos_pred = pred_v[u_local, pos_idx]
                    neg_pred = pred_v[u_local, neg_idx]

                    # Step 8.8.3) Weight by confidence at sampled positive
                    w_pos = S_v[u_local, pos_idx]

                    # Step 8.8.4) BPR sum: -sum( w_pos * log sigmoid(pos - neg) )
                    bpr_sum = -torch.sum(w_pos * F.logsigmoid(pos_pred - neg_pred))

                    total_bpr_sum += float(bpr_sum.item())
                    num_pairs += int(pos_pred.numel())
                else:
                    bpr_sum = torch.tensor(0.0, device=self.device)

                # Step 8.9) If neither term exists, skip update
                if (not has_obs) and (not valid.any()):
                    continue

                # Step 8.10) Total loss = SSE_obs + gamma * BPR_sum
                loss = sse_obs + (self.gamma * bpr_sum)

                # Step 8.11) Backprop + update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Step 8.12) Logging
            if (epoch == 0) or ((epoch + 1) % int(verbose_every) == 0):
                print(
                    f"Epoch {epoch+1}/{epochs}, "
                    f"SSE(obs): {total_sse_obs:.6f} (n_obs={num_obs}), "
                    f"BPR(sum): {total_bpr_sum:.6f} (n_pairs={num_pairs}), "
                    f"Total: {(total_sse_obs + self.gamma * total_bpr_sum):.6f}"
                )

    # Step 9) Full prediction for all users and return learned parameters
    def predict(self):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            users = torch.arange(self.user_count, device=self.device)
            pred = self(users).detach().cpu().numpy()

            params = {
                "p_u": self.P.weight.detach().cpu().numpy(),
                "q_i": self.Q.weight.detach().cpu().numpy(),
                "b_u": self.b_u.weight.detach().cpu().numpy(),
                "b_i": self.b_i.weight.detach().cpu().numpy(),
                "mu": self.mu.detach().cpu().numpy(),
            }
            return {"predictions": pred, "parameters": params}