from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout, GRU
from keras.optimizers import SGD

# Define the LSTM model
def old_running_lstm(epochs, X_train, y_train):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(rate=0.1))
    model.add(Dense(1, activation="linear"))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, shuffle=False, validation_split=0.2)

    return history, model

def run_LSTM(epochs, X_train, y_train):
    # The LSTM architecture
    modelLSTM = Sequential()
    # First LSTM layer with Dropout regularisation
    modelLSTM.add(Bidirectional(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2]))))
    modelLSTM.add(Dropout(0.2))
    # Second LSTM layer
    modelLSTM.add(Bidirectional(LSTM(units=50, return_sequences=True)))
    modelLSTM.add(Dropout(0.2))
    # Third LSTM layer
    modelLSTM.add(Bidirectional(LSTM(units=50, return_sequences=True)))
    modelLSTM.add(Dropout(0.2))
    # Fourth LSTM layer
    modelLSTM.add(Bidirectional(LSTM(units=50)))
    modelLSTM.add(Dropout(0.2))
    # The output layer
    modelLSTM.add(Dense(units=1))
    # Compiling the RNN
    modelLSTM.compile(optimizer='rmsprop',loss='mean_squared_error', metrics=['mse'])
    # Fitting to the training set
    history = modelLSTM.fit(X_train,y_train,epochs=epochs,batch_size=32,validation_split=0.2,shuffle=False)

    return history, modelLSTM


def run_GRU(epochs, X_train, y_train):
    # The GRU architecture
    modelGRU = Sequential()
    # First GRU layer with Dropout regularisation
    modelGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2]), activation='tanh'))
    modelGRU.add(Dropout(0.2))
    # Second GRU layer
    modelGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2]), activation='tanh'))
    modelGRU.add(Dropout(0.2))
    # Third GRU layer
    modelGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2]), activation='tanh'))
    modelGRU.add(Dropout(0.2))
    # Fourth GRU layer
    modelGRU.add(GRU(units=50, activation='tanh'))
    modelGRU.add(Dropout(0.2))
    # The output layer
    modelGRU.add(Dense(units=1))
    # Compiling the RNN
    modelGRU.compile(optimizer='SGD',loss='mean_squared_error', metrics=['mse'])
    # Fitting to the training set
    history = modelGRU.fit(X_train,y_train,epochs=epochs,batch_size=150,validation_split=0.2, shuffle=False)

    return history, modelGRU