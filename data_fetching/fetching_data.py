"""
F1 Data Fetcher: 2025 Season up to Imola
- Laps, results, weather
- Driver/Constructor standings
- Includes: FP1, FP2, FP3, Quali, Sprint, Race

Author: sid
Created: May 19 2025 10:33:16
"""

import os
import time
import fastf1
from fastf1 import ergast, Cache
import warnings
import pandas as pd

# === Setup ===
warnings.filterwarnings("ignore")
BASE_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/data_fetching"
CACHE_PATH = os.path.join(BASE_PATH, "f1_cache")
os.makedirs(CACHE_PATH, exist_ok=True)
Cache.enable_cache(CACHE_PATH)

# === Add Ergast standings ===
def save_standings(year, round_number, folder):
    try:
        driver_standings = ergast.fetch_driver_standings(year=year, round=round_number)
        constructor_standings = ergast.fetch_constructor_standings(year=year, round=round_number)

        if driver_standings is not None:
            driver_standings.to_csv(os.path.join(folder, "driver_standings.csv"), index=False)
        if constructor_standings is not None:
            constructor_standings.to_csv(os.path.join(folder, "constructor_standings.csv"), index=False)
    except Exception as e:
        print(f"⚠️ Standings fetch failed: {e}")

# === Fetch core session data ===
def download_full_session(year, event_name, session_type):
    try:
        print(f"📦 {year} {event_name} - {session_type}...", end="")
        session = fastf1.get_session(year, event_name, session_type)
        session.load()

        if session.laps.empty:
            print("⚠️ No driver data.")
            return

        round_number = session.event['RoundNumber']
        folder_name = f"{year}_{event_name.replace(' ', '_')}_{session_type}"
        folder = os.path.join(BASE_PATH, folder_name)
        os.makedirs(folder, exist_ok=True)

        session.laps.to_csv(os.path.join(folder, "laps.csv"), index=False)
        if session.results is not None:
            session.results.to_csv(os.path.join(folder, "results.csv"), index=False)
        if session.weather_data is not None:
            session.weather_data.to_csv(os.path.join(folder, "weather.csv"), index=False)

        save_standings(year, round_number, folder)

        print("✅ Saved")
        time.sleep(1.5)
    except Exception as e:
        print(f"❌ Failed: {e}")

# === Fetch 2025 data up to Imola ===
def fetch_2025_to_imola():
    year = 2025
    event_schedule = fastf1.get_event_schedule(year)
    
    for _, event in event_schedule.iterrows():
        event_name = event['EventName']
        round_number = event['RoundNumber']

        if round_number > 7:  # Imola is round 7 typically
            break

        for session_type in ['FP1', 'FP2', 'FP3', 'Q', 'S', 'R']:
            try:
                download_full_session(year, event_name, session_type)
            except Exception:
                continue

if __name__ == "__main__":
    fetch_2025_to_imola()