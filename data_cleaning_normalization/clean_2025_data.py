"""
FP-Friendly Cleaner for 2025 Season (Up to Imola)
- Includes FP1–FP3, Q, S, R sessions
- No DNF filtering
- Cleans lap data and merges weather
- Stops at round 7 (Imola)

Author: sid
Created : Tue May 20 2025  10:00:00
"""
import os
import pandas as pd
import numpy as np
import fastf1
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache("/Users/sid/Downloads/F1_FuturePrediction_2025/data_fetching/f1_cache")

RAW_DATA_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/data_fetching"
SAVE_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data/2025_cleaned.csv"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

def load_csv_safe(folder, filename):
    path = os.path.join(folder, filename)
    return pd.read_csv(path) if os.path.exists(path) else None

def normalize_columns(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

def clean_2025_data_fp_friendly():
    all_laps = []
    YEAR = 2025

    event_schedule = fastf1.get_event_schedule(YEAR)
    for _, event in event_schedule.iterrows():
        event_name = event["EventName"]
        round_number = event["RoundNumber"]

        if round_number > 7:  # Stop at Imola
            break

        for session_type in ['FP1', 'FP2', 'FP3', 'Q', 'S', 'R']:
            folder_name = f"{YEAR}_{event_name.replace(' ', '_')}_{session_type}"
            folder = os.path.join(RAW_DATA_PATH, folder_name)
            laps = load_csv_safe(folder, "laps.csv")
            weather = load_csv_safe(folder, "weather.csv")

            if laps is None or laps.empty:
                continue

            laps = laps.dropna(subset=['LapTime', 'DriverNumber'])
            if 'IsAccurate' in laps.columns:
                laps = laps[laps['IsAccurate'] == True]

            laps['DriverNumber'] = laps['DriverNumber'].astype(str)
            laps['Session'] = session_type
            gp_name = event_name.replace("Grand Prix", "Grand_Prix")
            laps['GP'] = gp_name

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

    if not all_laps:
        print(" No valid sessions found.")
        return

    df = pd.concat(all_laps, ignore_index=True)

    for col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']:
        if col in df.columns:
            try:
                df[col] = pd.to_timedelta(df[col]).dt.total_seconds()
            except:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'LapTime' in df.columns:
        df = remove_outliers_iqr(df, 'LapTime')

    numeric_cols = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
                    'AirTemp', 'TrackTemp', 'Humidity']
    existing = [col for col in numeric_cols if col in df.columns]
    df = normalize_columns(df, existing)

    df.to_csv(SAVE_PATH, index=False)
    print(f"✅ Saved: {SAVE_PATH}")

if __name__ == "__main__":
    clean_2025_data_fp_friendly()