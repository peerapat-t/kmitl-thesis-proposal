import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Step 1) Dataset: each batch returns (user_idx, Y[user_idx])
class PrepareDataset_Standard(Dataset):
    def __init__(self, Y_tensor: torch.Tensor):
        self.Y = Y_tensor

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, user_idx: int):
        return user_idx, self.Y[user_idx]


# Step 2) Standard MF
class Standard_Bias_MF(nn.Module):
    def __init__(self, user_count, item_count, K, learning_rate, lambda_rate, seed=42):
        super().__init__()
        torch.manual_seed(seed)

        self.user_count = int(user_count)
        self.item_count = int(item_count)
        self.K = int(K)
        self.learning_rate = float(learning_rate)
        self.lambda_rate = float(lambda_rate)

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

    # Step 3) Forward calculation (for a batch of users): produce scores for all items
    def forward(self, user_idx: torch.LongTensor) -> torch.Tensor:
        # Step 3.1) Gather user embeddings and user biases for the batch
        Pu = self.P(user_idx)
        bu = self.b_u(user_idx)

        # Step 3.2) Use all item embeddings and all item biases (shared for every user)
        Qt = self.Q.weight
        bi = self.b_i.weight.view(1, -1)

        # Step 3.3) Compute dot-products for all items and add biases
        return self.mu + bu + bi + (Pu @ Qt.T)

    # Step 4) Training loop using observed-only SSE
    def fit(self, Y: np.ndarray, epochs: int, batch_size: int = 512, verbose_every: int = 10):
        # Step 4.1) Validate input
        Y = np.asarray(Y, dtype=np.float32)
        if Y.ndim != 2 or Y.shape != (self.user_count, self.item_count):
            raise ValueError(f"Y must be shape {(self.user_count, self.item_count)}")
        if not np.isfinite(Y).all():
            raise ValueError("Y has NaN/Inf.")
        if Y.min() < 0:
            raise ValueError("Y has negative values.")

        # Step 4.2) Move model to device and set train mode
        self.to(self.device)
        self.train()

        # Step 4.3) Create tensor (kept on CPU; batches moved to device inside the loop)
        Y_tensor = torch.from_numpy(Y)

        # Step 4.4) Initialize global bias mu from mean of observed entries (Y != 0)
        if (Y_tensor != 0).any():
            with torch.no_grad():
                self.mu.fill_(float(Y_tensor[Y_tensor != 0].mean().item()))

        # Step 4.5) DataLoader over users
        loader = DataLoader(
            PrepareDataset_Standard(Y_tensor),
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=0,
        )

        # Step 4.6) Optimizer (L2 reg via weight_decay for embeddings only)
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

        # Step 5) Epoch loop
        for epoch in range(int(epochs)):
            total_sse = 0.0
            num_obs = 0

            # Step 5.1) Batch loop (per-user rows)
            for user_idx, Y_batch in loader:
                # Step 5.2) Move batch to device
                user_idx = user_idx.to(self.device)
                Y_batch = Y_batch.to(self.device)

                # Step 5.3) Predict all items for these users
                pred = self(user_idx)

                # Step 5.4) Build observed mask and skip if this batch has no observed entries
                mask = (Y_batch != 0)
                if not mask.any():
                    continue

                # Step 5.5) Compute SSE on observed entries only
                err = Y_batch[mask] - pred[mask]
                loss = torch.sum(err * err)

                # Step 5.6) Backprop and update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Step 5.7) Accumulate metrics
                total_sse += float(loss.item())
                num_obs += int(mask.sum().item())

            # Step 5.8) Logging
            if (epoch == 0) or ((epoch + 1) % int(verbose_every) == 0):
                print(f"Epoch {epoch+1}/{epochs}, SSE(obs): {total_sse:.6f} (n_obs={num_obs})")

    # Step 6) Predict full matrix and return parameters
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