"""
Created on Sat May 24 08:50:53 2025

@author: sid
Extract Monaco Fp1/Fp2 data
"""
import fastf1
from fastf1 import Cache
import os

# Setup local cache
BASE = "/Users/sid/Downloads/F1_FuturePrediction_2025/data_fetching"
CACHE_DIR = os.path.join(BASE, "f1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
Cache.enable_cache(CACHE_DIR)

# Fetch FP1 & FP2 for Monaco 2025
for session_type in ["FP1", "FP2"]:
    try:
        session = fastf1.get_session(2025, "Monaco Grand Prix", session_type)
        session.load()
        print(f"✅ Loaded {session_type}: {len(session.laps)} laps")

        folder = os.path.join(BASE, f"2025_Monaco_Grand_Prix_{session_type}")
        os.makedirs(folder, exist_ok=True)
        session.laps.to_csv(os.path.join(folder, "laps.csv"), index=False)
        if session.weather_data is not None:
            session.weather_data.to_csv(os.path.join(folder, "weather.csv"), index=False)
        if session.results is not None:
            session.results.to_csv(os.path.join(folder, "results.csv"), index=False)
    except Exception as e:
        print(f"❌ Failed {session_type}: {e}")