""" 
Created on Fri May 23 10:11:15 2025
@author: sid

Train Model Using Only Race & Quali Sessions
- Predicts final race position from Q + R sessions
- Uses LightGBM, XGBoost, RandomForest, SVR, LinearRegression
- Evaluates via MAE, R²
- Saves models, comparison table, and MAE bar plot
"""
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt

# === Paths ===
DATA_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data/training_data_race_quali.csv"
SAVE_DIR = "/Users/sid/Downloads/F1_FuturePrediction_2025/models_race_quali_cleaned/"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Load & Clean ===
df = pd.read_csv(DATA_PATH, low_memory=False) 

# Drop columns with no value
drop_cols = ['PitOutTime', 'PitInTime', 'DeletedReason', 'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

# Clean Rainfall and convert to numeric (assume 0 if empty or 'None')
df['Rainfall'] = pd.to_numeric(df['Rainfall'], errors='coerce').fillna(0)

# Drop remaining rows without Position
df = df.dropna(subset=['Position'])

# Drop non-feature columns
EXCLUDE = ['Time', 'LapStartTime', 'LapStartDate', 'Deleted', 'FastF1Generated',
           'Session', 'GP', 'DriverNumber', 'Position', 'Driver']
TARGET = 'Position'
categorical = ['Compound', 'Team']
numeric = [col for col in df.columns if col not in EXCLUDE + categorical]

# === Encode categoricals ===
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(df[categorical])
encoded_cat_cols = encoder.get_feature_names_out(categorical)
df_encoded = pd.DataFrame(encoded_cats, columns=encoded_cat_cols, index=df.index)

# === Final X and y ===
X = pd.concat([df[numeric], df_encoded], axis=1).fillna(0)
y = df[TARGET]

# === Split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Define Models ===
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.2),
    'LightGBM': lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
}

# === Train and Evaluate ===
results = []

for name, model in models.items():
    print(f"🚗 Training {name}")
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    print(f"🔍 {name} | MAE: {mae:.3f} | R²: {r2:.3f}")
    results.append((name, mae, r2))

    joblib.dump(model, os.path.join(SAVE_DIR, f"{name}_predictor.pkl"))

# === Save Results ===
results_df = pd.DataFrame(results, columns=["Model", "MAE", "R2"]).sort_values(by="MAE")
results_df.to_csv(os.path.join(SAVE_DIR, "model_eval_results.csv"), index=False)

# === Plot MAE
plt.figure(figsize=(10, 6))
plt.barh(results_df['Model'], results_df['MAE'], color='seagreen')
plt.xlabel("Mean Absolute Error")
plt.title("Model Comparison (Race + Quali Cleaned)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "mae_race_quali_cleaned.png"))
print("📊 Model evaluation results saved.")