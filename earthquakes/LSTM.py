# remove tenserflow info from console
#https://github.com/hayriyeanill/EarthquakePrediction/tree/main

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from math import sqrt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Bidirectional, Masking
from keras.layers import Dropout
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler



#plt.rcParams["figure.figsize"] = (15,5)
pd.options.mode.chained_assignment = None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')



"""
f_columns = ['year', 'month', 'day', 'day_of_week'] + numerical_cols

f_transformer = RobustScaler()
cnt_transformer = RobustScaler()

f_transformer = f_transformer.fit(train[f_columns].to_numpy())
cnt_transformer = cnt_transformer.fit(train[['magnitude']])

train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
train['magnitude'] = cnt_transformer.transform(train[['magnitude']])

test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
test['magnitude'] = cnt_transformer.transform(test[['magnitude']])
"""



def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 15

# reshape to [samples, time_steps, n_features]
X_train, Y_train = create_dataset(train, train['magnitude'], time_steps)
X_test, Y_test = create_dataset(test, test['magnitude'], time_steps)
print(f"Shape X train: {X_train.shape}, Shape Y train: {Y_train.shape}")
print()


model = Sequential()

#Layer for NaN Values
#model.add(Masking(mask_value=-1.0, input_shape=(X_train.shape[1], X_train.shape[2])))

model.add(Bidirectional(LSTM(units=16, input_shape=(X_train.shape[1], X_train.shape[2]))))
# 'relu' or 'tanh', 'linear'

model.add(Dropout(rate=0.1))
#add more layers here (dense or dropout)

model.add(Dense(units=1))


model.compile(loss='mean_squared_error', optimizer="adam", metrics=['mse'])
#optimizers 'RMSprop' or 'SGD', 'adam.

# grid search or random search to find the best combination of hyperparameters.
history = model.fit(X_train, Y_train, epochs=100, batch_size=64, validation_split=0.2, shuffle=False)


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

y_pred = model.predict(X_test)
y_pred_inv = cnt_transformer.inverse_transform(y_pred)

Y_test_inv = cnt_transformer.inverse_transform(Y_test.reshape(1, -1))
Y_train_inv = cnt_transformer.inverse_transform(Y_train.reshape(1, -1))

# convert Y_test and y_pred to dataframe
y_df = pd.DataFrame(Y_test_inv.T, columns=['Y_test_mag'])
y_df['y_pred_mag'] = y_pred_inv


"""
plt.plot(np.arange(0, len(Y_train)), Y_train_inv.flatten(), 'g', label="history")
plt.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)), Y_test_inv.flatten(), marker='.', label="true")
plt.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)), y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('magnitude')
plt.xlabel('Time Step')
plt.legend()
plt.show()
"""

plt.plot(Y_test_inv.flatten(), marker='.', label="true")
plt.plot(y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('magnitude')
plt.xlabel('Time Step')
plt.legend()
plt.show()

# Evaluation
rmse_lstm_pred = sqrt(mean_squared_error(y_df['Y_test_mag'], y_df['y_pred_mag']))
print("LSTM RMSE", rmse_lstm_pred)
lstm_score = r2_score(y_df['Y_test_mag'], y_df['y_pred_mag'])
print("LSTM r2 score ", lstm_score)
lstm_ms = mean_squared_error(y_df['Y_test_mag'], y_df['y_pred_mag'])
print("LSTM mean squared error ", lstm_ms)
lstm_m = mean_absolute_error(y_df['Y_test_mag'], y_df['y_pred_mag'])
print("LSTM mean absolute error ", lstm_m)
