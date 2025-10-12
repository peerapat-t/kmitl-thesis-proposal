import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class PrepareDataset(Dataset):
    def __init__(self, Y_tensor):
        self.Y = Y_tensor
    def __len__(self):
        return self.Y.shape[0]
    def __getitem__(self, user_idx):
        return user_idx, self.Y[user_idx]

class Standard_MF(torch.nn.Module):
    def __init__(self, K, learning_rate, lambda_rate):
        super().__init__()
        torch.manual_seed(42)
        self.K = K
        self.learning_rate = learning_rate
        self.lambda_rate = lambda_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.P_embedding, self.Q_embedding = None, None

    def _initialize_parameters(self, Y):
        user_count, item_count = Y.shape
        self.user_count, self.item_count = user_count, item_count
        self.P_embedding = torch.nn.Embedding(user_count, self.K)
        self.Q_embedding = torch.nn.Embedding(item_count, self.K)
        nn.init.normal_(self.P_embedding.weight, std=0.01)
        nn.init.normal_(self.Q_embedding.weight, std=0.01)

    def forward(self, user_indices):
        p_u = self.P_embedding(user_indices)
        q_i = self.Q_embedding.weight
        prediction = p_u @ q_i.T
        return prediction

    def fit(self, Y, epochs, batch_size=32):
        if self.P_embedding is None: self._initialize_parameters(Y)
        self.to(self.device)
        self.train()
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        num_observed_ratings = torch.sum(Y_tensor > 0).item()
        dataset = PrepareDataset(Y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        all_params = [self.P_embedding.weight, self.Q_embedding.weight]
        optimizer = torch.optim.SGD(all_params, lr=self.learning_rate, weight_decay=self.lambda_rate)
        for epoch in range(epochs):
            total_loss = 0
            for user_indices, Y_batch in dataloader:
                optimizer.zero_grad()
                user_indices, Y_batch = user_indices.to(self.device), Y_batch.to(self.device)
                y_hat = self(user_indices)
                mask = Y_batch > 0
                error = (Y_batch[mask] - y_hat[mask])
                loss = (error**2).sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                avg_loss = np.sqrt(total_loss / num_observed_ratings)
                print(f"Epoch {epoch+1}/{epochs}, Train RMSE: {avg_loss:.4f}")

    def predict(self):
        if self.P_embedding is None:
            raise RuntimeError("Model has not been trained. Please call fit() first.")
        self.eval()
        with torch.no_grad():
            all_user_indices = torch.arange(self.user_count, device=self.device)
            predictions = self(all_user_indices)
            parameters = {
                'p': self.P_embedding.weight.detach().cpu().numpy(),
                'q': self.Q_embedding.weight.detach().cpu().numpy()
            }
            return {
                'predictions': predictions.detach().cpu().numpy(),
                'parameters': parameters
            }