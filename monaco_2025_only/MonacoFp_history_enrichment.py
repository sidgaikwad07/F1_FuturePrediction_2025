"""
Created on Sat May 24 15:13:56 2025

@author: sid

"""
import pandas as pd

# === Paths ===
FP_DATA_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data/monaco_2025_fp_cleaned.csv"
OUTPUT_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data/monaco_2025_fp_enriched.csv"

# === Load Data ===
df = pd.read_csv(FP_DATA_PATH)
print(f"📦 Monaco FP data shape: {df.shape}")

# === DriverNumber → DriverName Mapping ===
driver_map = {
    1: 'Max Verstappen',
    4: 'Lando Norris',
    5: 'Gabriel Bortoleto',
    6: 'Isack Hadjar',
    10: 'Pierre Gasly',
    12: 'Kimi Antonelli',
    14: 'Fernando Alonso',
    16: 'Charles Leclerc',
    18: 'Lance Stroll',
    22: 'Yuki Tsunoda',
    23: 'Alexander Albon',
    27: 'Nico Hülkenberg',
    30: 'Liam Lawson',
    31: 'Esteban Ocon',
    43: 'Franco Colapinto',
    44: 'Lewis Hamilton',
    55: 'Carlos Sainz',
    63: 'George Russell',
    81: 'Oscar Piastri',
    87: 'Oliver Bearman'
}

df['DriverName'] = df['DriverNumber'].map(driver_map)

# === Monaco Historical Driver Score (based on real-world + reputation) ===
monaco_history_scores = {
    'Max Verstappen': 9.8,
    'Charles Leclerc': 9.7,
    'Lewis Hamilton': 9.3,
    'Lando Norris': 9.1,
    'Carlos Sainz': 9.0,
    'Fernando Alonso': 9.0,
    'George Russell': 8.7,
    'Oscar Piastri': 8.5,
    'Sergio Perez': 8.0,
    'Pierre Gasly': 7.5,
    'Esteban Ocon': 7.4,
    'Alexander Albon': 7.3,
    'Nico Hülkenberg': 7.2,
    'Lance Stroll': 6.9,
    'Yuki Tsunoda': 6.8,
    'Oliver Bearman': 5.5,
    'Kimi Antonelli': 5.0,
    'Franco Colapinto': 5.0,
    'Isack Hadjar': 5.0,
    'Liam Lawson': 6.5,
    'Gabriel Bortoleto': 5.0
}

df['MonacoHistoryScore'] = df['DriverName'].map(monaco_history_scores)

# === Save Enriched Monaco FP Data ===
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved enriched FP data:\n{OUTPUT_PATH}")