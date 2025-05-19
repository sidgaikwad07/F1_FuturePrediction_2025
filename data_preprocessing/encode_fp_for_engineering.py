"""
Encode FP Data for Advanced Feature Engineering
- Prepares data for driver/team/track-based Monaco-aware features
- Includes stable driver IDs, encoded categories, session metadata
- Filters to FP1, FP2, FP3
- Merges Race result for supervision

Author: sid
Updated: Mon 19 May 2025 14:05:00
"""

import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# === Paths ===
INPUT_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data/merged_cleaned_2021_2025.csv"
OUTPUT_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data/encoded_fp_for_engineering.csv"

# === Load merged cleaned data
df = pd.read_csv(INPUT_PATH)
print(f"📂 Loaded dataset with {df.shape[0]} rows")

# === Normalize GP & Driver for matching
for col in ['GP', 'Driver']:
    df[col] = df[col].astype(str).str.strip().str.lower().str.replace(" ", "_")

# === Create stable DriverID from name
unique_drivers = sorted(df['Driver'].dropna().unique())
driver_id_map = {name: idx for idx, name in enumerate(unique_drivers)}
df['DriverID'] = df['Driver'].map(driver_id_map)

# === Session encoding
session_map = {'FP1': 0, 'FP2': 1, 'FP3': 2, 'Q': 3, 'S': 4, 'R': 5}
df['SessionEncoded'] = df['Session'].map(session_map)

# === Monaco/Street track flag
df['IsStreetCircuit'] = df['GP'].str.contains("monaco|singapore|jeddah|baku|vegas", case=False).astype(int)
df['IsMonaco'] = df['GP'].str.contains("monaco", case=False).astype(int)

# === Ordinal encode
ordinal_cols = ['DriverID', 'Team', 'Compound', 'GP']
df[ordinal_cols] = df[ordinal_cols].astype(str)
encoder = OrdinalEncoder()
encoded_array = encoder.fit_transform(df[ordinal_cols])
encoded_df = pd.DataFrame(encoded_array, columns=[f"{col}_Encoded" for col in ordinal_cols])
df = pd.concat([df, encoded_df], axis=1)

# === Extract FP sessions
fp_df = df[df['Session'].isin(['FP1', 'FP2', 'FP3'])].copy()

# === Extract Race result
race_df = df[df['Session'] == 'R'][['Season', 'GP', 'Driver', 'Position']].copy()
race_df = race_df.drop_duplicates(subset=['Season', 'GP', 'Driver'])

# === Get latest FP session per driver per GP
fp_latest = fp_df.groupby(['Season', 'GP', 'Driver']).last().reset_index()

# === Safe merge with race result
merged_df = fp_latest.merge(race_df, on=['Season', 'GP', 'Driver'], how='left')

# === Save output
merged_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Final encoded dataset saved to:\n{OUTPUT_PATH}")
print(f"🔢 Total rows: {merged_df.shape[0]} | Columns: {merged_df.shape[1]}")