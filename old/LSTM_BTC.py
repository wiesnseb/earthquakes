#https://medium.com/codex/time-series-prediction-using-lstm-in-python-19b1187f580f

import ccxt
import numpy as np
import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


ex = ccxt.binance()

# download data from binance spot market
df = pd.DataFrame(
    ex.fetch_ohlcv(symbol='BTCUSDT', timeframe='1d', limit=1000), 
    columns = ['unix', 'open', 'high', 'low', 'close', 'volume']
)

# convert unix (in milliseconds) to UTC time
df['date'] = pd.to_datetime(df.unix, unit='ms')


scaler = MinMaxScaler()
# fit the format of the scaler -> convert shape from (1000, ) -> (1000, 1)
close_price = df.close.values.reshape(-1, 1)
scaled_close = scaler.fit_transform(close_price)

seq_len = 60

def split_into_sequences(data, seq_len):
    n_seq = len(data) - seq_len + 1
    return np.array([data[i:(i+seq_len)] for i in range(n_seq)])

def get_train_test_sets(data, seq_len, train_frac):
    sequences = split_into_sequences(data, seq_len)
    n_train = int(sequences.shape[0] * train_frac)
    x_train = sequences[:n_train, :-1, :]
    y_train = sequences[:n_train, -1, :]
    x_test = sequences[n_train:, :-1, :]
    y_test = sequences[n_train:, -1, :]
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = get_train_test_sets(scaled_close, seq_len, train_frac=0.9)

# fraction of the input to drop; helps prevent overfitting
dropout = 0.1
window_size = seq_len - 1

# build a 3-layer LSTM RNN
model = keras.Sequential()

model.add(
    LSTM(window_size, return_sequences=True, 
         input_shape=(window_size, x_train.shape[-1]))
)

model.add(Dropout(rate=dropout))
# Bidirectional allows for training of sequence data forwards and backwards
model.add(
    Bidirectional(LSTM((window_size * 2), return_sequences=True)
)) 

model.add(Dropout(rate=dropout))
model.add(
    Bidirectional(LSTM(window_size, return_sequences=False))
) 

model.add(Dense(units=1))

# linear activation function: activation is proportional to the input
model.add(Activation('linear'))

batch_size = 16

model.compile(
    loss='mean_squared_error',
    optimizer='adam'
)

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=batch_size,
    shuffle=False,
    validation_split=0.2
)

y_pred = model.predict(x_test)

# invert the scaler to get the absolute price data
y_test_orig = scaler.inverse_transform(y_test)
y_pred_orig = scaler.inverse_transform(y_pred)

# plots of prediction against actual data
plt.figure(figsize=(15,5))
plt.plot(y_test_orig, label='Actual Price', color='red')
plt.plot(y_pred_orig, label='Predicted Price', color='blue')
plt.title('BTC Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend(loc='best')
plt.show()