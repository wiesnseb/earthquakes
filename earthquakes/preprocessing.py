#import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def split_data(df, split_ratio):
    train_data, test_data = train_test_split(df, test_size=split_ratio, random_state=42)
    return train_data, test_data


def normalize_columns(df, columns):
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[columns] = scaler.fit_transform(df[columns])
    return df_normalized


def create_dataset(df, feature_cols, target_col, time_steps=1):
    Xs, ys = [], []
    for i in range(len(df) - time_steps):
        Xs.append(df[feature_cols].iloc[i:(i + time_steps)].values)
        ys.append(df[target_col].iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


def fit_scaler(df, target_variable):
    scaler = MinMaxScaler()
    fitted_scaler = scaler.fit(df[[target_variable]])
    
    return fitted_scaler
"""
#Old create dataset function from initial code
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)
"""

"""
def split_data(df, ratio):
    train_size = int(len(df) * ratio)
    train = df.iloc[0:train_size]
    test = df.iloc[train_size:len(df)]
    
    return train, test
"""