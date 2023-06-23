from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import pandas as pd


def evaluate_model(y_true, y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    ms = mean_squared_error(y_true, y_pred)
    m = mean_absolute_error(y_true, y_pred)
    print("RMSE:", rmse)
    print("R^2 score:", r2)
    print("Mean squared error:", ms)
    print("Mean absolute error:", m)


from sklearn.preprocessing import MinMaxScaler

def get_predictions(model, X_test, y_test, target_scaler):
    y_pred = model.predict(X_test)
    y_pred_inv = target_scaler.inverse_transform(y_pred)
    y_test_inv = target_scaler.inverse_transform(y_test.reshape(1, -1))
    #y_train_inv = target_scaler.inverse_transform(y_train.reshape(1, -1))
    
    result_df = pd.DataFrame(y_test_inv.T, columns=['y_test_mag'])
    result_df['y_pred_mag'] = y_pred_inv

    return result_df