"""
Created on Fri May 23 09:42:41 2025

@author: sid
Driver & Team Historical Form Feature Engineering
- Computes avg race rank, DNF rate, and qualifying position for last 3 GPs
- Merges with Monaco-aware FP features
"""

import os
import pandas as pd
import numpy as np
import warnings as warnings

MERGED_CLEAN_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data/merged_cleaned_2021_2025.csv"
FP_FEATURE_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data/encoded_fp_for_engineering.csv"
OUTPUT_PATH = "/Users/sid/Downloads/F1_FuturePrediction_2025/clean_data/final_fp_with_history.csv"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# === STEP 1: Compute historical stats from Q and R sessions ===
def compute_driver_history(merged):
    # Race session processing
    race_df = merged[merged['Session'] == 'R'].copy()
    race_df['DNF_flag'] = race_df['Position'].isna().astype(int)
    race_summary = race_df.groupby(['Year', 'GP', 'DriverNumber']).agg({
        'Position': 'min',
        'DNF_flag': 'max'    
    }).reset_index()
    race_summary.sort_values(['DriverNumber', 'Year', 'GP'], inplace=True)
    race_summary['avg_finish_last_3'] = race_summary.groupby('DriverNumber')['Position']\
                                                     .rolling(3, min_periods=1).mean().reset_index(drop=True)
    race_summary['dnf_rate_last_3'] = race_summary.groupby('DriverNumber')['DNF_flag']\
                                                   .rolling(3, min_periods=1).mean().reset_index(drop=True)
    hist_race_features = race_summary[['Year', 'GP', 'DriverNumber', 'avg_finish_last_3', 'dnf_rate_last_3']]

    # Qualifying session processing
    qual_df = merged[merged['Session'] == 'Q'].copy()
    qual_summary = qual_df.groupby(['Year', 'GP', 'DriverNumber'])['Position'].min().reset_index()
    qual_summary.sort_values(['DriverNumber', 'Year', 'GP'], inplace=True)
    qual_summary['avg_qual_pos_last_3'] = qual_summary.groupby('DriverNumber')['Position']\
                                                       .rolling(3, min_periods=1).mean().reset_index(drop=True)

    hist_qual_features = qual_summary[['Year', 'GP', 'DriverNumber', 'avg_qual_pos_last_3']]
    return hist_race_features, hist_qual_features

# === STEP 2: Merge onto FP features ===
def main():
    merged = pd.read_csv(MERGED_CLEAN_PATH)
    fp = pd.read_csv(FP_FEATURE_PATH)
    race_feats, qual_feats = compute_driver_history(merged)
    df = fp.merge(race_feats, on=['Year', 'GP', 'DriverNumber'], how='left')
    df = df.merge(qual_feats, on=['Year', 'GP', 'DriverNumber'], how='left')
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Final model-ready feature dataset saved:\n{OUTPUT_PATH}")

if __name__ == "__main__":
    main()