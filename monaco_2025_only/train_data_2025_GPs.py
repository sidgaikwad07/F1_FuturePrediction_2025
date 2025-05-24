"""
Created on Sat May 24 14:35:25 2025

@author: sid
Builds Training Dataset Using 2025 GPs Only (Race + Quali)
"""
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# === Paths ===
MERGED_FILE = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data/merged_cleaned_2021_2025.csv"
SAVE_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data/training_data_2025_only.csv"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# === Load Data ===
df = pd.read_csv(MERGED_FILE, low_memory=False)

# Filter for 2025 Race + Quali sessions
df = df[(df['Year'] == 2025) & (df['Session'].isin(['R', 'Q']))]
df = df.dropna(subset=['Position'])

# Drop non-feature columns
drop_cols = ['PitOutTime', 'PitInTime', 'DeletedReason', 'Sector1SessionTime', 
             'Sector2SessionTime', 'Sector3SessionTime', 'Time', 'LapStartTime', 
             'LapStartDate', 'Deleted', 'FastF1Generated', 'Session', 'GP', 
             'DriverNumber', 'Driver']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Rainfall conversion
df['Rainfall'] = pd.to_numeric(df['Rainfall'], errors='coerce').fillna(0)

# One-hot encoding
categorical = ['Compound', 'Team']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[categorical])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical), index=df.index)

# Final feature matrix
df = pd.concat([df.drop(columns=categorical), encoded_df], axis=1).fillna(0)
df.to_csv(SAVE_PATH, index=False)
print(f"✅ Training dataset saved to:\n{SAVE_PATH}")