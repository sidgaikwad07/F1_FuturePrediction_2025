"""
Created on Sat May 24 15:29:21 2025

@author: sid

"""
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder

# === Paths ===
FP_DATA_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data/monaco_2025_fp_enriched.csv"
MODEL_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/models_2025_only/XGBoost_2025.pkl"
SAVE_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/predictions/top10_monaco_2025_predictions.csv"

# === Load Data
df = pd.read_csv(FP_DATA_PATH)
print(f"📦 Monaco FP data shape: {df.shape}")

df = df.dropna(subset=["LapTime", "DriverNumber"])

# === Column groups
categorical = ['Compound', 'Team']
exclude = ['DriverNumber', 'DriverName', 'Time', 'LapStartTime', 'LapStartDate', 'Deleted', 'FastF1Generated']
numeric = [col for col in df.columns if col not in categorical + exclude]

# === One-Hot Encode
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[categorical])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical), index=df.index)

X = pd.concat([df[numeric], encoded_df], axis=1)

# === Align with model
model = joblib.load(MODEL_PATH)
model_features = model.feature_names_in_

# Ensure all model features are present
X = X.reindex(columns=model_features, fill_value=0)

# === Predict
df['PredictedPosition'] = model.predict(X)

# === Monaco-aware driver modifiers (OPTIONAL)
monaco_modifier = {
    "Charles Leclerc": -0.4,
    "Yuki Tsunoda": +0.3,
    "Franco Colapinto": +0.5,
    "Lando Norris": -0.2,
    "Lewis Hamilton": -0.2,
    "Oscar Piastri": -0.1,
    "Carlos Sainz": -0.2,
    "Max Verstappen": 0.0
}
df['PredictedPosition'] += df['DriverName'].map(monaco_modifier).fillna(0)

# === Summarize per driver (best lap prediction)
summary = (
    df.groupby(['DriverNumber', 'DriverName', 'Team'])['PredictedPosition']
    .min()
    .reset_index()
    .sort_values('PredictedPosition')
    .reset_index(drop=True)
)
summary["FinalPredictedPosition"] = summary.index + 1

# === Output Top 10
top10 = summary.head(10)[['FinalPredictedPosition', 'DriverName', 'Team', 'PredictedPosition']]

print("\n✅ Top 10 Monaco 2025 Predictions:\n")
print(top10.to_string(index=False))

top10.to_csv(SAVE_PATH, index=False)
print(f"\n📁 Saved to: {SAVE_PATH}")