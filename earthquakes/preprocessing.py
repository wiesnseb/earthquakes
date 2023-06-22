import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def split_data(df, ratio):
    train_size = int(len(df) * ratio)
    test_size = len(df) - train_size
    train = df.iloc[0:train_size]
    test = df.iloc[train_size:len(df)]
    
    print(f"Length dataset: {len(df)}")
    print(f"Length training set: {len(train)}")
    print(f"Length test set: {len(test)}")
    
    return train, test


def normalize_columns(df, columns):
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[columns] = scaler.fit_transform(df[columns])
    return df_normalized