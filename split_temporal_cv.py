import pandas as pd
import numpy as np

def create_temporal_folds_with_cold_start(df, n_splits=5):

    unique_user_ids = sorted(df['userid'].unique())
    unique_item_ids = sorted(df['itemid'].unique())

    user_to_idx = {user_id: i for i, user_id in enumerate(unique_user_ids)}
    item_to_idx = {item_id: i for i, item_id in enumerate(unique_item_ids)}

    df['user_idx'] = df['userid'].map(user_to_idx)
    df['item_idx'] = df['itemid'].map(item_to_idx)

    n_users = len(unique_user_ids)
    n_items = len(unique_item_ids)

    df_sorted = df.sort_values(by='timestamp').reset_index(drop=True)

    chunks = np.array_split(df_sorted, n_splits + 1)

    final_df_folds = []

    for i in range(1, n_splits + 1):
        train_df = pd.concat(chunks[:i])
        test_df = chunks[i].copy()
        final_df_folds.append((train_df, test_df))

    train_arrays_list = []
    test_arrays_list = []
    cold_start_flag_list = []

    for (train_df, test_df) in final_df_folds:
        train_matrix = np.zeros((n_users, n_items))
        test_matrix = np.zeros((n_users, n_items))

        for row in train_df.itertuples(index=False):
            train_matrix[row.user_idx, row.item_idx] = row.rating

        for row in test_df.itertuples(index=False):
            test_matrix[row.user_idx, row.item_idx] = row.rating

        train_arrays_list.append(train_matrix)
        test_arrays_list.append(test_matrix)
        
        user_cold_flags = np.ones(n_users, dtype=bool)
        train_user_indices = train_df['user_idx'].unique()
        if len(train_user_indices) > 0:
            user_cold_flags[train_user_indices] = False
        
        cold_start_flag_list.append(user_cold_flags)

    return train_arrays_list, test_arrays_list, cold_start_flag_list