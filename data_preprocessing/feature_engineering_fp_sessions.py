"""
Feature Engineering Script for FP Sessions (2021–2025)
- Builds Monaco-aware features from FP1 and FP2
- Includes rookie detection, teammate delta, lap stats
- Weather integration, compound encoding

Author: sid
Updated: Fri May 23 2025 (Bugfix for OneHotEncoder)
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder

# === CONFIG ===
MERGED_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data/merged_cleaned_2021_2025.csv"
SAVE_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data/encoded_fp_for_engineering.csv"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# === STEP 1: Filter FP sessions ===
def load_and_filter_fp_sessions(path):
    df = pd.read_csv(path, low_memory=False)
    return df[df['Session'].isin(['FP1', 'FP2'])].copy()

# === STEP 2: Rolling average of last 3 laps ===
def compute_rolling_avg_lap(df):
    df.sort_values(['Year', 'GP', 'DriverNumber', 'Time'], inplace=True)
    df['rolling_avg_3_laps'] = df.groupby(['Year', 'GP', 'DriverNumber'])['LapTime']\
                                 .rolling(3, min_periods=1).mean().reset_index(level=[0,1,2], drop=True)
    return df

# === STEP 3: Position delta (lap to lap) ===
def compute_position_delta(df):
    df['position_delta'] = df.groupby(['Year', 'GP', 'DriverNumber'])['Position']\
                              .diff().fillna(0)
    return df

# === STEP 4: Detect rookies (no history before 2025) ===
def detect_rookies(df):
    first_seen = df.groupby('DriverNumber')['Year'].min().reset_index()
    first_seen['is_rookie'] = (first_seen['Year'] == 2025).astype(int)
    return df.merge(first_seen[['DriverNumber', 'is_rookie']], on='DriverNumber', how='left')

# === STEP 5: Delta to teammate in FP2 ===
def teammate_delta_fp2(df):
    fp2 = df[df['Session'] == 'FP2']
    avg_laps = fp2.groupby(['Year', 'GP', 'DriverNumber'])['LapTime'].mean().reset_index()
    teams = fp2[['Year', 'GP', 'DriverNumber', 'Team']].drop_duplicates()
    merged = avg_laps.merge(teams, on=['Year', 'GP', 'DriverNumber'], how='left')
    teammate_mean = merged.groupby(['Year', 'GP', 'Team'])['LapTime'].transform('mean')
    merged['delta_to_teammate_fp2'] = merged['LapTime'] - teammate_mean
    return df.merge(merged[['Year', 'GP', 'DriverNumber', 'delta_to_teammate_fp2']],
                    on=['Year', 'GP', 'DriverNumber'], how='left')

# === STEP 6: Encode tyre compound ===
def encode_compounds(df):
    if 'Compound' not in df.columns:
        return df
    df['Compound'] = df['Compound'].fillna('Unknown')
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    compounds = encoder.fit_transform(df[['Compound']])
    compound_labels = encoder.get_feature_names_out(['Compound'])
    encoded_df = pd.DataFrame(compounds, columns=compound_labels, index=df.index)
    return pd.concat([df, encoded_df], axis=1)

# === STEP 7: Weather features ===
def compute_weather_flags(df):
    df['track_temp_avg'] = df.groupby(['Year', 'GP', 'Session'])['TrackTemp'].transform('mean')
    df['humidity_avg'] = df.groupby(['Year', 'GP', 'Session'])['Humidity'].transform('mean')
    df['rain_flag'] = (df['Rainfall'].fillna(0) > 0).astype(int)
    return df

# === STEP 8: Final cleanup ===
def finalize_features(df):
    df = df.dropna(subset=['LapTime', 'DriverNumber'])
    if 'Compound' in df.columns:
        df.drop(columns=['Compound'], inplace=True)
    return df

# === MAIN ===
def main():
    df = load_and_filter_fp_sessions(MERGED_PATH)
    print(f" Loaded dataset with {len(df)} FP1/FP2 rows")

    df = compute_rolling_avg_lap(df)
    df = compute_position_delta(df)
    df = detect_rookies(df)
    df = teammate_delta_fp2(df)
    df = encode_compounds(df)
    df = compute_weather_flags(df)
    df = finalize_features(df)

    df.to_csv(SAVE_PATH, index=False)
    print(f"✅ Final encoded dataset saved:\n{SAVE_PATH}")

if __name__ == "__main__":
    main()