"""
Created on Sat May 24 14:50:42 2025

@author: sid
Feature Engineer Monaco 2025 FP1/FP2
- Adds rolling average, weather flags, compound encoding
- Flags rookies, teammate deltas, etc.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os

# === Paths ===
INPUT_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data/monaco_2025_fp_cleaned.csv"
OUTPUT_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data/monaco_2025_fp_engineered.csv"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# === Load
df = pd.read_csv(INPUT_PATH)
print(f"📥 Loaded Monaco FP shape: {df.shape}")

# === Rolling average (last 3 laps)
df = df.sort_values(['DriverNumber', 'LapNumber'])
df['rolling_avg_3_laps'] = df.groupby('DriverNumber')['LapTime'].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)

# === Rookie flag
rookies = [5, 6, 12, 30, 43, 87]  # DriverNumbers for rookies
df['is_rookie'] = df['DriverNumber'].isin(rookies).astype(int)

# === Teammate delta (FP2 only)
fp2 = df[df['Session'] == 'FP2']
avg_lap = fp2.groupby(['DriverNumber'])['LapTime'].mean()
team_map = fp2[['DriverNumber', 'Team']].drop_duplicates().set_index('DriverNumber')['Team'].to_dict()
df['Team'] = df['DriverNumber'].map(team_map)
team_avg = fp2.groupby('Team')['LapTime'].mean()
df['delta_to_teammate_fp2'] = df['LapTime'] - df['Team'].map(team_avg)

# === Weather summary
df['track_temp_avg'] = df.groupby(['Session'])['TrackTemp'].transform('mean')
df['humidity_avg'] = df.groupby(['Session'])['Humidity'].transform('mean')
df['rain_flag'] = (df['Rainfall'] > 0).astype(int)

# === Compound encoding
df['Compound'] = df['Compound'].fillna("Unknown")
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[['Compound']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Compound']), index=df.index)
df = pd.concat([df, encoded_df], axis=1)
df.drop(columns=['Compound'], inplace=True)

# === Final save
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved engineered Monaco FP to:\n{OUTPUT_PATH}")