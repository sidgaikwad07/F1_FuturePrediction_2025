"""
Clean & Normalize 2021 F1 Season Data
- Merges laps, weather, results
- Removes DNFs, invalid laps
- Treats outliers and normalizes features

Author: sid
Updated: 17 May 2025
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# === CONFIG ===
RAW_DATA_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/data_fetching"
SAVE_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data/2021_cleaned.csv"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# === Helper Functions ===

def load_csv_safe(folder, filename):
    path = os.path.join(folder, filename)
    return pd.read_csv(path) if os.path.exists(path) else None

def is_dnf(driver_number, results_df):
    try:
        status = results_df.loc[results_df['DriverNumber'] == int(driver_number), 'Status'].values
        return 'Finished' not in str(status)
    except:
        return True

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

def normalize_columns(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

# === Main Processing ===
def clean_2021_data():
    all_laps = []
    YEAR = 2021

    for folder_name in tqdm(os.listdir(RAW_DATA_PATH), desc="🧹 Cleaning 2021"):
        if not folder_name.startswith(str(YEAR)):
            continue

        folder = os.path.join(RAW_DATA_PATH, folder_name)

        laps = load_csv_safe(folder, "laps.csv")
        results = load_csv_safe(folder, "results.csv")
        weather = load_csv_safe(folder, "weather.csv")

        if laps is None or results is None:
            continue

        # Remove laps with NaT or missing data
        laps = laps.dropna(subset=['LapTime', 'DriverNumber'])
        if 'IsAccurate' in laps.columns:
            laps = laps[laps['IsAccurate'] == True]

        # Remove DNFs
        laps['DriverNumber'] = laps['DriverNumber'].astype(str)
        laps = laps[~laps['DriverNumber'].apply(lambda x: is_dnf(x, results))]

        # Add session info
        laps['Session'] = folder_name.split("_")[-1]
        laps['GP'] = "_".join(folder_name.split("_")[1:-1])

        # Merge with weather if available
        if weather is not None and 'Time' in weather.columns and not weather['Time'].isnull().all():
            try:
                weather['Time'] = pd.to_timedelta(weather['Time'])
                laps['Time'] = pd.to_timedelta(laps['Time'])
                laps = pd.merge_asof(
                    laps.sort_values('Time'),
                    weather.sort_values('Time'),
                    on='Time',
                    direction='nearest'
                )
            except Exception as e:
                print(f"⚠️ Weather merge failed for {folder_name}: {e}")

        all_laps.append(laps)

    # === Concatenate all session laps
    df = pd.concat(all_laps, ignore_index=True)

    # === Convert time columns to numeric BEFORE outlier removal
    time_cols = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']
    for col in time_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_timedelta(df[col]).dt.total_seconds()
            except Exception as e:
                print(f"⛔ Could not convert {col} to timedelta: {e}")
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # === Remove outliers (IQR)
    for col in time_cols:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)

    # === Normalize numeric columns
    numeric_cols = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
                    'AirTemp', 'TrackTemp', 'Humidity']
    existing_cols = [c for c in numeric_cols if c in df.columns]
    df = normalize_columns(df, existing_cols)

    # === Save cleaned version
    df.to_csv(SAVE_PATH, index=False)
    print(f"\n✅ Cleaned 2021 season saved to:\n{SAVE_PATH}")

if __name__ == "__main__":
    clean_2021_data()