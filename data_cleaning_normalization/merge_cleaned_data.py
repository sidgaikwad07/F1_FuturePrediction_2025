"""
Created on Mon May 19 11:00:04 2025

@author: sid
Combine the yearly features from the year (2021-2025)
"""
import os
import pandas as pd

# === CONFIG ===
CLEANED_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data"
YEARS = [2021, 2022, 2023, 2024, 2025]
SAVE_PATH = os.path.join(CLEANED_PATH, "merged_cleaned_2021_2025.csv")

# === Load and concatenate
def merge_all_cleaned_years():
    all_dfs = []

    for year in YEARS:
        file_path = os.path.join(CLEANED_PATH, f"{year}_cleaned.csv")
        if os.path.exists(file_path):
            print(f"📂 Loading {year}_cleaned.csv...")
            df = pd.read_csv(file_path)
            df['Season'] = year  # Add season indicator
            all_dfs.append(df)
        else:
            print(f" Missing file for {year}: {file_path}")

    if not all_dfs:
        print("⛔ No cleaned datasets found. Check file paths.")
        return

    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df.to_csv(SAVE_PATH, index=False)
    print(f"\n✅ Merged dataset saved to:\n{SAVE_PATH}")
    print(f"🔢 Total rows: {len(merged_df)} | Total columns: {merged_df.shape[1]}")

if __name__ == "__main__":
    merge_all_cleaned_years()
