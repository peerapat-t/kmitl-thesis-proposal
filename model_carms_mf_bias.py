import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")

class PrepareDataset(Dataset):
    
    def __init__(self, Y_tensor, S_tensor):
        self.Y = Y_tensor
        self.S = S_tensor

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, user_idx):
        return user_idx, self.Y[user_idx], self.S[user_idx]

class CARMS_MF_Bias(torch.nn.Module):
    
    def __init__(self, user_count, item_count, K, learning_rate, lambda_rate, gamma):

        super().__init__()
        torch.manual_seed(42)
        
        self.user_count = user_count
        self.item_count = item_count
        self.K = K
        self.learning_rate = learning_rate
        self.lambda_rate = lambda_rate
        self.gamma = gamma

        self.P_embedding = torch.nn.Embedding(self.user_count, self.K)
        self.Q_embedding = torch.nn.Embedding(self.item_count, self.K)
        self.R_embedding = torch.nn.Embedding(self.item_count, self.K)
        
        self.user_bias = torch.nn.Embedding(self.user_count, 1)
        self.item_bias = torch.nn.Embedding(self.item_count, 1)
        self.global_bias = torch.nn.Parameter(torch.tensor(0.0))

        nn.init.normal_(self.P_embedding.weight, std=0.01)
        nn.init.normal_(self.Q_embedding.weight, std=0.01)
        nn.init.normal_(self.R_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loss_fn = nn.MSELoss(reduction='sum')

    def forward(self, user_indices):

        user_factors = self.P_embedding(user_indices)
        item_factors_y = self.Q_embedding.weight
        item_factors_s = self.R_embedding.weight
        
        user_b = self.user_bias(user_indices)
        item_b = self.item_bias.weight
        
        y_prediction = (user_factors @ item_factors_y.T) + user_b + item_b.T + self.global_bias
        s_prediction = (user_factors @ item_factors_s.T)
        
        return y_prediction, s_prediction

    def fit(self, Y, S, epochs, batch_size=32):

        self.to(self.device)
        self.train()

        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        S_tensor = torch.tensor(S, dtype=torch.float32)

        observed_mask_y = Y_tensor > 0
        observed_mask_s = S_tensor != 0
        
        num_observed_ratings = torch.sum(observed_mask_y).item()
        num_observed_s = torch.sum(observed_mask_s).item()

        if num_observed_ratings == 0:
            print("Warning: No observed ratings (Y > 0) found in the dataset.")

        dataset = PrepareDataset(Y_tensor, S_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        params_with_decay = [
            self.P_embedding.weight,
            self.Q_embedding.weight,
            self.R_embedding.weight
        ]
        
        params_without_decay = [
            self.user_bias.weight,
            self.item_bias.weight,
            self.global_bias
        ] 

        optimizer = torch.optim.Adam(
            [
                {'params': params_with_decay, 'weight_decay': self.lambda_rate},
                {'params': params_without_decay, 'weight_decay': 0.0}
            ],
            lr=self.learning_rate
        )
        
        for epoch in range(epochs):
            total_loss_y = 0
            total_loss_s = 0
            
            for user_indices, Y_batch, S_batch in dataloader:

                user_indices = user_indices.to(self.device)
                Y_batch = Y_batch.to(self.device)
                S_batch = S_batch.to(self.device)
                
                y_hat, s_hat = self(user_indices)
                
                mask_y = Y_batch > 0
                y_true_observed = Y_batch[mask_y]
                y_pred_observed = y_hat[mask_y]
                loss_y = self.loss_fn(y_pred_observed, y_true_observed)
                
                mask_s = S_batch != 0
                s_true_observed = S_batch[mask_s]
                s_pred_observed = s_hat[mask_s]
                loss_s = self.loss_fn(s_pred_observed, s_true_observed)
                
                loss = loss_y + (self.gamma * loss_s)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0) 
                optimizer.step()
                
                total_loss_y += loss_y.item()
                total_loss_s += loss_s.item()
            
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                rmse_y = np.sqrt(total_loss_y / num_observed_ratings) if num_observed_ratings > 0 else 0
                rmse_s = np.sqrt(total_loss_s / num_observed_s) if num_observed_s > 0 else 0
                print(f"Epoch {epoch+1}/{epochs}, Train RMSE (Y): {rmse_y:.4f}, Train RMSE (S): {rmse_s:.4f}")

    def predict(self):

        self.eval()
        
        with torch.no_grad():

            all_user_indices = torch.arange(self.user_count, device=self.device)
            
            y_pred, s_pred = self(all_user_indices)
            
            parameters = {
                'p_u': self.P_embedding.weight.detach().cpu().numpy(),
                'q_i': self.Q_embedding.weight.detach().cpu().numpy(),
                'r_i': self.R_embedding.weight.detach().cpu().numpy(),
                'b_u': self.user_bias.weight.detach().cpu().numpy(),
                'b_i': self.item_bias.weight.detach().cpu().numpy(),
                'mu': self.global_bias.detach().cpu().numpy()
            }
            
            return {
                'predictions': y_pred.detach().cpu().numpy(),
                'parameters': parameters
            }