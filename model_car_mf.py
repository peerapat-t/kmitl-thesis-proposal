import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import warnings
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.types import StructType, StructField, ArrayType, StringType

warnings.filterwarnings("ignore")


def generate_signals(Y, k_user=30, k_item=20, min_support=0.3, min_confidence=0.5):
    spark = SparkSession.builder.appName("FP-growth Signals").getOrCreate()

    num_users, num_items = Y.shape

    user_labels = KMeans(n_clusters=k_user, random_state=42, n_init=10).fit_predict(Y)
    item_labels = KMeans(n_clusters=k_item, random_state=42, n_init=10).fit_predict(Y.T)

    transactions_binarized = (Y >= 4)

    transaction_list = []
    for u_idx in range(num_users):
        trans = {f'user_group_{user_labels[u_idx]}'}
        liked_items = np.where(transactions_binarized[u_idx])[0]
        for i_idx in liked_items:
            trans.add(f'item_{i_idx}')
            trans.add(f'item_group_{item_labels[i_idx]}')
        if len(liked_items) > 0:
            transaction_list.append(list(trans))

    if not transaction_list:
        spark.stop()
        return np.zeros_like(Y), np.zeros_like(Y)

    schema = StructType([StructField("items", ArrayType(StringType()))])
    transaction_df = spark.createDataFrame(
        [(transaction,) for transaction in transaction_list],
        schema=schema
    )

    fp_growth = FPGrowth(
        itemsCol="items",
        minSupport=min_support,
        minConfidence=min_confidence
    )
    model = fp_growth.fit(transaction_df)
    
    rules = model.associationRules.toPandas()

    if rules.empty:
        spark.stop()
        return np.zeros_like(Y), np.zeros_like(Y)

    rules['antecedents'] = rules['antecedent'].apply(frozenset)
    rules['consequents'] = rules['consequent'].apply(frozenset)
    
    rules = rules[~rules['consequents'].apply(lambda x: any(str(i).startswith('user_group_') for i in x))]
    if rules.empty:
        spark.stop()
        return np.zeros_like(Y), np.zeros_like(Y)
    
    matrix_item_signal = np.zeros_like(Y, dtype=float)
    matrix_group_item_signal = np.zeros_like(Y, dtype=float)
    item_map = {f'item_{i}': i for i in range(num_items)}
    item_group_map = {f'item_group_{i}': i for i in range(k_item)}
    
    item_to_users = defaultdict(set)
    for user_idx, history in enumerate(transaction_list):
        for item in history:
            item_to_users[item].add(user_idx)

    for _, rule in rules.iterrows():
        antecedents = list(rule['antecedents'])
        if not antecedents: continue
        
        matching_users = item_to_users.get(antecedents[0], set()).copy()
        for item in antecedents[1:]:
            matching_users.intersection_update(item_to_users.get(item, set()))
        
        if not matching_users: continue

        target_user_indices = list(matching_users)
        for consequent in rule['consequents']:
            if consequent in item_map:
                col_idx = item_map[consequent]
                matrix_item_signal[target_user_indices, col_idx] += rule['confidence']
            elif consequent in item_group_map:
                target_items = np.where(item_labels == item_group_map[consequent])[0]
                rows, cols = np.meshgrid(target_user_indices, target_items)
                matrix_group_item_signal[rows, cols] += rule['confidence']
    
    if np.any(matrix_item_signal):
        matrix_item_signal = MinMaxScaler().fit_transform(matrix_item_signal)
    if np.any(matrix_group_item_signal):
        matrix_group_item_signal = MinMaxScaler().fit_transform(matrix_group_item_signal)
    
    spark.stop()
    
    return matrix_item_signal, matrix_group_item_signal


class PrepareDataset(Dataset):
    def __init__(self, Y_tensor):
        self.Y = Y_tensor

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, user_idx):
        return user_idx, self.Y[user_idx]


class CAR_MF(torch.nn.Module):
    def __init__(self, K, learning_rate, lambda_rate, k_user=30, k_item=20, min_support=0.3, min_confidence=0.5):
        super().__init__()
        torch.manual_seed(42)
        self.K = K
        self.learning_rate = learning_rate
        self.lambda_rate = lambda_rate
        self.k_user = k_user
        self.k_item = k_item
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.P_embedding, self.Q_embedding = None, None
        self.Alpha_embedding, self.Beta_embedding = None, None
        self.M_tensor, self.N_tensor = None, None

    def _initialize_parameters(self, Y):
        user_count, item_count = Y.shape
        self.user_count, self.item_count = user_count, item_count
        self.P_embedding = torch.nn.Embedding(user_count, self.K)
        self.Q_embedding = torch.nn.Embedding(item_count, self.K)
        self.Alpha_embedding = torch.nn.Embedding(user_count, 1)
        self.Beta_embedding = torch.nn.Embedding(user_count, 1)

        nn.init.normal_(self.P_embedding.weight, std=0.01)
        nn.init.normal_(self.Q_embedding.weight, std=0.01)
        nn.init.normal_(self.Alpha_embedding.weight, std=0.01)
        nn.init.normal_(self.Beta_embedding.weight, std=0.01)

    def forward(self, user_indices):
        p_u = self.P_embedding(user_indices)
        q_i = self.Q_embedding.weight
        alpha_u = self.Alpha_embedding(user_indices)
        beta_u = self.Beta_embedding(user_indices)
        m_ui = self.M_tensor[user_indices]
        n_ui = self.N_tensor[user_indices]
        
        prediction = (p_u @ q_i.T) + (alpha_u * m_ui) + (beta_u * n_ui)
        return prediction

    def fit(self, Y, epochs, batch_size=32):
        M, N = generate_signals(
            Y,
            k_user=self.k_user,
            k_item=self.k_item,
            min_support=self.min_support,
            min_confidence=self.min_confidence
        )

        if self.P_embedding is None: self._initialize_parameters(Y)
        self.to(self.device)
        self.train()

        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        self.M_tensor = torch.tensor(M, dtype=torch.float32).to(self.device)
        self.N_tensor = torch.tensor(N, dtype=torch.float32).to(self.device)
        num_observed_ratings = torch.sum(Y_tensor > 0).item()
        
        dataset = PrepareDataset(Y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        params_with_decay = [self.P_embedding.weight, self.Q_embedding.weight]
        params_without_decay = [self.Alpha_embedding.weight, self.Beta_embedding.weight]

        optimizer = torch.optim.Adam([
            {'params': params_with_decay, 'weight_decay': self.lambda_rate},
            {'params': params_without_decay, 'weight_decay': 0.0}
        ], lr=self.learning_rate)
                
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
                'q': self.Q_embedding.weight.detach().cpu().numpy(),
                'alphau': self.Alpha_embedding.weight.detach().cpu().numpy(),
                'betau': self.Beta_embedding.weight.detach().cpu().numpy()
            }
            return {
                'predictions': predictions.detach().cpu().numpy(),
                'parameters': parameters
            }