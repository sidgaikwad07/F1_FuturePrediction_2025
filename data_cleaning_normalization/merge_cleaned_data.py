"""
Created on Tue May 20 10:15:04 2025

@author: sid
Merge Cleaned F1 Data (2021–2025)
- Adds Year column
- Stacks all sessions including FP, Q, R, S
- Outputs full unified dataset

"""
import os
import pandas as pd

def merge_cleaned_data(input_dir, output_file, years=(2021, 2022, 2023, 2024, 2025)):
    """
    Merge cleaned F1 data across multiple years.

    Parameters:
    - input_dir: folder path containing individual year CSVs
    - output_file: full path to save merged CSV
    - years: tuple of years to merge
    """
    all_dfs = []

    for year in years:
        filename = f"{year}_cleaned.csv"
        path = os.path.join(input_dir, filename)
        if not os.path.exists(path):
            print(f"⚠️ Missing: {filename}")
            continue

        df = pd.read_csv(path, low_memory=False)
        df["Year"] = year
        all_dfs.append(df)
        print(f"✅ Loaded: {filename} ({len(df)} rows)")

    if not all_dfs:
        print(" No data loaded.")
        return

    merged_df = pd.concat(all_dfs, ignore_index=True)

    # Overview
    print("\n📊 Session Count:")
    print(merged_df["Session"].value_counts())

    merged_df.to_csv(output_file, index=False)
    print(f"\n✅ Merged dataset saved to:\n{output_file}")

# === Example Usage ===
if __name__ == "__main__":
    input_folder = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data"
    output_path = os.path.join(input_folder, "merged_cleaned_2021_2025.csv")
    merge_cleaned_data(input_folder, output_path)
