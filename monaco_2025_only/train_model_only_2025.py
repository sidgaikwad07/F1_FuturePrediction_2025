"""
Created on Sat May 24 14:37:19 2025

@author: sid
Train XGBoost Model on 2025 GPs Only (Race + Quali)
Evaluates with MAE, RMSE, R² and saves model and metrics.

"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# === Paths ===
DATA_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data/training_data_2025_only.csv"
MODEL_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/models_2025_only/XGBoost_2025.pkl"
PLOT_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/models_2025_only/feature_importance_2025.png"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# === Load data ===
df = pd.read_csv(DATA_PATH)
print(f"📦 Loaded training data: {df.shape}")

# === Split features and target ===
y = df['Position']
X = df.drop(columns=['Position'])

# === Train/test split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Model training ===
model = XGBRegressor(
    n_estimators=250,
    max_depth=6,
    learning_rate=0.07,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

print("🚗 Training XGBoost...")
model.fit(X_train, y_train)

# === Evaluation ===
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print("\n✅ Evaluation Results:")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")

# === Save model ===
joblib.dump(model, MODEL_PATH)
print(f"📁 Model saved to: {MODEL_PATH}")

# === Feature Importance Plot ===
plt.figure(figsize=(12, 6))
plot_importance(model, max_num_features=20, importance_type='gain', height=0.5)
plt.title("Top 20 Feature Importances (XGBoost - 2025 Only)")
plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"📊 Feature importance plot saved: {PLOT_PATH}")