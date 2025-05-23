"""
Created on Fri May 23 10:17:02 2025

@author: sid
Prepare Race + Qualifying Training Dataset
- Filters merged_cleaned_2021_2025.csv to include only R + Q sessions
- Drops rows with missing Position
- Saves cleaned training dataset
"""
import pandas as pd
import os

# === Paths ===
INPUT_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data/merged_cleaned_2021_2025.csv"
OUTPUT_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data/training_data_race_quali.csv"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# === Load merged dataset ===
df = pd.read_csv(INPUT_PATH, low_memory=False)

# === Filter R and Q sessions only ===
df_filtered = df[df['Session'].isin(['R', 'Q'])]

# === Drop rows without Position label ===
df_filtered = df_filtered.dropna(subset=['Position'])

print(f"✅ Final training dataset: {len(df_filtered)} rows")
df_filtered.to_csv(OUTPUT_PATH, index=False)
print(f"📁 Saved to: {OUTPUT_PATH}")
