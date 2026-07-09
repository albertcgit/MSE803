import os
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import load_and_preprocess
from models import train_linear, train_xgb, train_ann, train_lstm, train_arima
from evaluate import evaluate
from sklearn.model_selection import train_test_split

# ✅ Create folders
os.makedirs("outputs/plots", exist_ok=True)

# ✅ Load data
X, X_scaled, y = load_and_preprocess("airline-passengers.csv", "target_column")

# ✅ Train-test split (tabular models)
X_train, X_test, Xs_train, Xs_test, y_train, y_test = train_test_split(
    X, X_scaled, y, test_size=0.2, random_state=42
)

results = []

# =============================
# ✅ Linear Regression
# =============================
lr_model = train_linear(Xs_train, y_train)
lr_pred = lr_model.predict(Xs_test)

lr_metrics = evaluate(y_test, lr_pred)
lr_metrics["Model"] = "Linear Regression"
results.append(lr_metrics)


# =============================
# ✅ XGBoost
# =============================
xgb_model = train_xgb(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

xgb_metrics = evaluate(y_test, xgb_pred)
xgb_metrics["Model"] = "XGBoost"
results.append(xgb_metrics)


# =============================
# ✅ ANN
# =============================
ann_model = train_ann(Xs_train, y_train)
ann_pred = ann_model.predict(Xs_test).flatten()

ann_metrics = evaluate(y_test, ann_pred)
ann_metrics["Model"] = "ANN"
results.append(ann_metrics)


# =============================
# ✅ LSTM
# =============================
lstm_model, X_lstm = train_lstm(X_scaled, y)

split = int(len(X_lstm) * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
y_train_lstm, y_test_lstm = y[:split], y[split:]

lstm_model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, verbose=0)

y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()

lstm_metrics = evaluate(y_test_lstm, y_pred_lstm)
lstm_metrics["Model"] = "LSTM"
results.append(lstm_metrics)


# =============================
# ✅ ARIMA
# =============================
series = y.reset_index(drop=True)

train_size = int(len(series) * 0.8)
train_arima = series[:train_size]
test_arima = series[train_size:]

arima_model = train_arima(train_arima)

forecast = arima_model.forecast(steps=len(test_arima))

arima_metrics = evaluate(test_arima, forecast)
arima_metrics["Model"] = "ARIMA"
results.append(arima_metrics)


# =============================
# ✅ SAVE RESULTS
# =============================
results_df = pd.DataFrame(results)

# Save results table
results_df.to_csv("outputs/results.csv", index=False)

print("\n✅ Model Results:\n", results_df)


# =============================
# ✅ SAVE PREDICTIONS
# =============================
predictions_df = pd.DataFrame({
    "Actual": y_test.reset_index(drop=True),
    "Linear Regression": lr_pred,
    "XGBoost": xgb_pred,
    "ANN": ann_pred
})

predictions_df.to_csv("outputs/model_predictions.csv", index=False)


# =============================
# ✅ PLOTS
# =============================

# Model comparison bar chart
results_df.set_index("Model")[["RMSE", "MAE", "MAPE"]].plot(kind="bar", figsize=(10,6))
plt.title("Model Comparison")
plt.ylabel("Error")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/plots/model_comparison.png")
plt.close()


# Actual vs predicted (XGBoost)
plt.figure(figsize=(6,6))
plt.scatter(y_test, xgb_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (XGBoost)")
plt.tight_layout()
plt.savefig("outputs/plots/xgboost_actual_vs_pred.png")
plt.close()


print("\n✅ All outputs saved in 'outputs/' folder")