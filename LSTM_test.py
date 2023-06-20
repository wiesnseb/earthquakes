# remove tenserflow info from console
#https://github.com/hayriyeanill/EarthquakePrediction/tree/main

import os
from math import sqrt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Bidirectional
from keras.layers import Dropout
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
plt.rcParams["figure.figsize"] = (15,5)


#2019-01-12 04:08:37
df = pd.read_csv("Earthquake_db.csv",  delimiter=',', parse_dates=['Date'])

#023-02-17 09:37:34.868000+00:00
#df = pd.read_csv("data/earthquakes1900_2023.csv", delimiter=',', parse_dates=['Date'])
#df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

df = df.set_index('Date')


train_size = int(len(df) * 0.75)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(f"Number of rows training set: {len(train)}, number of rows test set: {len(test)}")
print()

f_columns = ['Latitude', 'Longitude', 'Depth']

f_transformer = RobustScaler()
cnt_transformer = RobustScaler()

f_transformer = f_transformer.fit(train[f_columns].to_numpy())
cnt_transformer = cnt_transformer.fit(train[['Magnitude']])

train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
train['Magnitude'] = cnt_transformer.transform(train[['Magnitude']])

test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
test['Magnitude'] = cnt_transformer.transform(test[['Magnitude']])


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 15

# reshape to [samples, time_steps, n_features]
X_train, y_train = create_dataset(train, train['Magnitude'], time_steps)
X_test, y_test = create_dataset(test, test['Magnitude'], time_steps)
print(f"Shape X train: {X_train.shape}, Shape Y train: {y_train.shape}")
print()


model = Sequential()
model.add(Bidirectional(LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2]))))
model.add(Dropout(rate=0.2))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train,epochs=30, batch_size=32, validation_split=0.1, shuffle=False)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

y_pred = model.predict(X_test)
y_pred_inv = cnt_transformer.inverse_transform(y_pred)

y_test_inv = cnt_transformer.inverse_transform(y_test.reshape(1, -1))
y_train_inv = cnt_transformer.inverse_transform(y_train.reshape(1, -1))

# convert y_test and y_pred to dataframe
y_df = pd.DataFrame(y_test_inv.T, columns=['y_test_mag'])
y_df['y_pred_mag'] = y_pred_inv


"""
plt.plot(np.arange(0, len(y_train)), y_train_inv.flatten(), 'g', label="history")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test_inv.flatten(), marker='.', label="true")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Magnitude')
plt.xlabel('Time Step')
plt.legend()
plt.show()
"""

plt.plot(y_test_inv.flatten(), marker='.', label="true")
plt.plot(y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Magnitude')
plt.xlabel('Time Step')
plt.legend()
plt.show()

# Evaluation
rmse_lstm_pred = sqrt(mean_squared_error(y_df['y_test_mag'], y_df['y_pred_mag']))
print("LSTM RMSE", rmse_lstm_pred)
lstm_score = r2_score(y_df['y_test_mag'], y_df['y_pred_mag'])
print("LSTM r2 score ", lstm_score)
lstm_ms = mean_squared_error(y_df['y_test_mag'], y_df['y_pred_mag'])
print("LSTM mean squared error ", lstm_ms)
lstm_m = mean_absolute_error(y_df['y_test_mag'], y_df['y_pred_mag'])
print("LSTM mean absolute error ", lstm_m)