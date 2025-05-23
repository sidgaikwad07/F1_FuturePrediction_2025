""" 
Created on Fri May 23 10:50:39 2025
@author: sid

Hyperparameter Tuning for XGBoost on Race+Quali Dataset
- Uses Optuna to minimize MAE
- Saves best model and trial results
"""
import pandas as pd
import numpy as np
import os
import optuna
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

# === Paths ===
DATA_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data/training_data_race_quali.csv"
SAVE_BEST_MODEL = "/Users/sid/Downloads/F1_FuturePrediction_2025/models_race_quali_cleaned/XGBoost_tuned.pkl"

# === Load and Preprocess ===
df = pd.read_csv(DATA_PATH, low_memory=False)

drop_cols = ['PitOutTime', 'PitInTime', 'DeletedReason', 'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')
df['Rainfall'] = pd.to_numeric(df['Rainfall'], errors='coerce').fillna(0)
df = df.dropna(subset=['Position'])

EXCLUDE = ['Time', 'LapStartTime', 'LapStartDate', 'Deleted', 'FastF1Generated', 'Session', 'GP', 'DriverNumber', 'Position', 'Driver']
TARGET = 'Position'
categorical = ['Compound', 'Team']
numeric = [col for col in df.columns if col not in EXCLUDE + categorical]

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[categorical])
cat_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical), index=df.index)

X = pd.concat([df[numeric], cat_df], axis=1).fillna(0)
y = df[TARGET]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Objective for Optuna ===
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "random_state": 42,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    return mae

# === Run Optuna Study ===
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

# === Save best model ===
best_params = study.best_params
print(f"\n✅ Best Params: {best_params}")

best_model = xgb.XGBRegressor(**best_params)
best_model.fit(X_train, y_train)
joblib.dump(best_model, SAVE_BEST_MODEL)
print(f"✅ Best model saved to:\n{SAVE_BEST_MODEL}")