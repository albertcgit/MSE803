from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# ✅ Linear Regression
def train_linear(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# ✅ XGBoost
def train_xgb(X_train, y_train):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    return model


# ✅ ANN
def train_ann(X_train, y_train):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    return model


# ✅ LSTM
def train_lstm(X_scaled, y):
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(1, X_scaled.shape[1])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    return model, X_lstm


# ✅ ARIMA
def train_arima(series):
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()
    return model_fit