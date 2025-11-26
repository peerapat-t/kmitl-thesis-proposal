import pandas as pd
import numpy as np
import warnings
from sklearn.cluster import KMeans
from collections import defaultdict
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

warnings.filterwarnings("ignore")

class SignalGenerator:
    
    def __init__(self, k_user=30, k_item=20, min_support=0.3, 
                 min_confidence=0.5, rating_percentile_threshold=0.8,
                 random_state=42):

        self.k_user = k_user
        self.k_item = k_item
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.rating_percentile_threshold = rating_percentile_threshold
        self.random_state = random_state
        
        self.user_labels_ = None
        self.item_labels_ = None
        self.transaction_list_ = None
        self.rules_ = None
        self.num_users_ = None
        self.num_items_ = None

    def fit_transform(self, Y: np.ndarray) -> np.ndarray:
        self.num_users_, self.num_items_ = Y.shape
        
        # Step 1: Clustering
        self.user_labels_, self.item_labels_ = self._run_clustering(Y)

        # Step 2: Binarization
        transactions_binarized = self._binarize_ratings(Y)

        # Step 3: Transaction List
        self.transaction_list_ = self._create_transaction_list(transactions_binarized)

        # Step 4: FPGrowth & Association Rules
        self.rules_ = self._find_association_rules()
        if self.rules_.empty:
            return np.zeros_like(Y, dtype=float) 
        
        # Step 5: Filter Association Rules
        self.rules_ = self._filter_rules()
        if self.rules_.empty:
            return np.zeros_like(Y, dtype=float) 
        
        # Step 6: Build Signal Matrix
        matrix_signal = self._build_signal_matrix(Y)
        
        # Step 7: Apply 1log
        norm_matrix_signal = np.log1p(matrix_signal)
        
        return norm_matrix_signal

    def _run_clustering(self, Y: np.ndarray):
        user_labels = KMeans(n_clusters=self.k_user, random_state=self.random_state, n_init=10).fit_predict(Y)
        item_labels = KMeans(n_clusters=self.k_item, random_state=self.random_state, n_init=10).fit_predict(Y.T)
        return user_labels, item_labels

    def _binarize_ratings(self, Y: np.ndarray) -> np.ndarray:
        Y_masked = np.where(Y > 0, Y, np.nan)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            min_ratings = np.nanmin(Y_masked, axis=1, keepdims=True)
            max_ratings = np.nanmax(Y_masked, axis=1, keepdims=True)
        
        min_ratings = np.nan_to_num(min_ratings, nan=0.0)
        max_ratings = np.nan_to_num(max_ratings, nan=1.0)
        max_ratings[max_ratings == min_ratings] = min_ratings[max_ratings == min_ratings] + 1.0 

        thresholds_per_user = min_ratings + (max_ratings - min_ratings) * self.rating_percentile_threshold
        
        return (Y >= thresholds_per_user)

    def _create_transaction_list(self, transactions_binarized: np.ndarray):
        transaction_list = []
        for u_idx in range(self.num_users_):
            trans = {f'user_group_{self.user_labels_[u_idx]}'}
            liked_items = np.where(transactions_binarized[u_idx])[0] 
            
            if len(liked_items) > 0:
                for i_idx in liked_items:
                    trans.add(f'item_{i_idx}')
                    trans.add(f'item_group_{self.item_labels_[i_idx]}')
                transaction_list.append(list(trans))
                
        return transaction_list

    def _find_association_rules(self):
        if not self.transaction_list_:
            return pd.DataFrame()
        
        te = TransactionEncoder()
        te_ary = te.fit(self.transaction_list_).transform(self.transaction_list_)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        
        frequent_itemsets = fpgrowth(df, min_support=self.min_support, use_colnames=True)
        if frequent_itemsets.empty:
            return pd.DataFrame()

        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)
        return rules

    def _filter_rules(self):
        if self.rules_.empty:
            return self.rules_
        filtered_rules = \
            self.rules_[~self.rules_['consequents'].apply(lambda x: any(str(i).startswith('user_group_') for i in x))]
        return filtered_rules

    def _build_signal_matrix(self, Y: np.ndarray):
        matrix_signal = np.zeros_like(Y, dtype=float)
        item_map = {f'item_{i}': i for i in range(self.num_items_)}
        item_group_map = {f'item_group_{i}': i for i in range(self.k_item)}
        
        item_to_users = defaultdict(set)
        for user_idx, history in enumerate(self.transaction_list_):
            for item in history:
                item_to_users[item].add(user_idx)

        for antecedents_set, group_df in self.rules_.groupby('antecedents'):
            antecedents = list(antecedents_set) 
            if not antecedents: continue
            
            try:
                matching_users = item_to_users.get(antecedents[0], set()).copy()
                for item in antecedents[1:]:
                    matching_users.intersection_update(item_to_users.get(item, set()))
            except Exception as e:
                continue
                
            if not matching_users: continue
            
            target_user_indices = list(matching_users)
            
            for _, rule in group_df.iterrows():
                for consequent in rule['consequents']:
                    if consequent in item_map:
                        col_idx = item_map[consequent] 
                        matrix_signal[target_user_indices, col_idx] += rule['confidence']
                    elif consequent in item_group_map:
                        target_items = np.where(self.item_labels_ == item_group_map[consequent])[0]
                        if target_items.size == 0: continue
                        rows, cols = np.meshgrid(target_user_indices, target_items, indexing='ij')
                        matrix_signal[rows, cols] += rule['confidence']

        return matrix_signal