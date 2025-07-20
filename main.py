import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import datetime
from fetch_n_save_data import raw_stock_data
from data_processing import processed_stock_data, sanitized_data_df
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from visualization_matplot import (stock_plot,
                                   plot_regression_overlay,
                                   plot_classification_overlay,
                                   plot_histogram,
                                   visualize_predictions)
import seaborn as sns


# # Use this code while using Streamlit only
# from stock_prediction_app import data_frm_streamlit, handle_stock_selection
# selected_stock,selected_stock_data, selected_full_name = data_frm_streamlit()
# user_selection, refresh_flag = handle_stock_selection(selected_stock, selected_stock_data, selected_full_name, today)
# raw_df = raw_stock_data(user_input=user_selection, refresh_stock_data=refresh_flag)


# -------------------------- 📅 Log Start Time --------------------------
time_now = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
today = datetime.date.today().strftime("%d-%m-%Y")
print(f'The date and time now is: {time_now}')

# -------------------------- 📂 Load and Clean Data --------------------------
raw_df = raw_stock_data(user_input='A', refresh_stock_data='n')  # Comment for Streamlit
processed_df = processed_stock_data(raw_df)
sanitized_df = sanitized_data_df(processed_df)
df = sanitized_df
print(df.head())

# -------------------------- 📈 Feature Engineering --------------------------
df['return'] = df['close'].pct_change()
df['volatility'] = df['close'].rolling(window=5).std()
df['sma_ratio'] = df['sma_5'] / df['sma_10']
df['volume_change'] = df['volume'].pct_change()

# Multi-day forward return for smoother signal classification
df['return_5d'] = df['close'].shift(-5) / df['close'] - 1

def classify_5d_signal(r):
    if r > 0.03: return 2  # Buy
    elif r < -0.03: return 0  # Sell
    else: return 1  # Hold

df['signal'] = df['return_5d'].apply(classify_5d_signal)

# Also include 1-day return for regression
df['future_return'] = df['close'].shift(-1) / df['close'] - 1

# -------------------------- 🎯 Modeling Prep --------------------------
feature_cols = ['return', 'volatility', 'sma_ratio', 'volume_change']
df = df.dropna(subset=feature_cols + ['signal', 'future_return'])

X = df[feature_cols]
y_class = df['signal']                # Classification target
y_reg = df['future_return']          # Regression target

# -------------------------- 🧠 Train Models --------------------------
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

clf = RandomForestClassifier()
clf.fit(X_train, y_class_train)

reg = Ridge()
reg.fit(X_train, y_reg_train)

# -------------------------- 🔮 Predict --------------------------
y_pred_class = clf.predict(X_test)
y_pred_reg = reg.predict(X_test)
confidences = clf.predict_proba(X_test).max(axis=1)

# -------------------------- 📦 Compile Output --------------------------
prediction_df = pd.DataFrame({
    "date": df.loc[X_test.index]['date'].values,
    "signal": y_pred_class,
    "expected_return": y_pred_reg,
    "confidence": confidences
})

# Strategic logic: combine classifier + regressor + confidence
def generate_trade_signal(row, return_threshold=0.01, confidence_threshold=0.6):
    if row["signal"] == 2 and row["expected_return"] > return_threshold and row["confidence"] > confidence_threshold:
        return "strong_buy"
    elif row["signal"] == 0 and row["expected_return"] < -return_threshold and row["confidence"] > confidence_threshold:
        return "strong_sell"
    elif row["signal"] == 1:
        return "weak_hold"
    else:
        return "ignore"

prediction_df["final_signal"] = prediction_df.apply(generate_trade_signal, axis=1)
print(prediction_df["final_signal"].value_counts())

# -------------------------- 🧭 Signal Mapping --------------------------
signal_map = {0: "sell", 1: "hold", 2: "buy"}
prediction_df['signal_label'] = prediction_df['signal'].map(signal_map)
prediction_df["class_match"] = (y_class_test == y_pred_class).astype(int)

# -------------------------- 📊 Evaluation Functions --------------------------
def evaluate_classification(y_true, y_pred, class_labels=None):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n📊 Classification Accuracy: {accuracy:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels) if class_labels else classification_report(y_true, y_pred))
    print("\n🔍 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    match_rate = (y_true == y_pred).mean() * 100
    print(f"\n✅ Sample-wise classification match rate: {match_rate:.2f}%")
    return accuracy, match_rate

def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print("\n📊 Regression MAE:", mae)
    print("📊 Regression MSE:", mse)
    print("📊 Regression RMSE:", rmse)
    print("📊 Regression R² Score:", r2)
    return mae, mse, rmse, r2

# Run evaluation
evaluate_classification(y_class_test, y_pred_class, class_labels=list(signal_map.values()))
evaluate_regression(y_reg_test, y_pred_reg)

# -------------------------- 📉 Error Computation --------------------------
def compute_regression_percent_error(y_true, y_pred):
    actual = pd.Series(y_true).replace(0, np.nan)
    predicted = pd.Series(y_pred)
    error = ((predicted - actual) / actual) * 100
    return error.replace([np.inf, -np.inf], np.nan).dropna()

reg_percent_error = compute_regression_percent_error(y_reg_test.values, y_pred_reg)


# -------------------------- 📊 Visualization --------------------------

# 📦 Batch metric visualization
plot_data_dict = {
    "Actual Future Return": y_reg_test.values,
    "Predicted Future Return": y_pred_reg,
    "Predicted Signal": y_pred_class,
    "Confidence Score": confidences
}
fig = stock_plot(
    datas=list(plot_data_dict.values()),
    column_names=list(plot_data_dict.keys())
)
plt.show()

# 📈 Overlay: Regression Prediction
plot_regression_overlay(
    y_actual=y_reg_test.values,
    y_predicted=y_pred_reg,
    title="Future Return",
    xlabel="Test Sample Index",
    ylabel="Return"
)

# 📈 Overlay: Signal Classification
plot_classification_overlay(
    y_actual=y_class_test,
    y_predicted=y_pred_class,
    class_map=signal_map,
    title="Signal",
    xlabel="Test Sample Index",
    ylabel="Signal"
)

# 📉 Regression % Error Distribution
plot_histogram(
    data=reg_percent_error,
    bins=50,
    title="Distribution of % Error (Regression Predictions)",
    xlabel="% Error",
    ylabel="Frequency",
    clip_range=(-100, 100)
)

# 🔍 Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(
    confusion_matrix(y_class_test, y_pred_class),
    annot=True,
    cmap="Blues",
    fmt='d',
    xticklabels=list(signal_map.values()),
    yticklabels=list(signal_map.values())
)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()