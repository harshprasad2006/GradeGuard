"""
Data Ingestion and Inspection Script for GradeGuard.
Fetches the official UCI dataset ID 697 and saves a clean local copy.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

RAW_DATA_PATH = os.path.join("data", "raw", "dataset.csv")

def fetch_and_save_data():
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    
    # Check if raw data already exists
    if os.path.exists(RAW_DATA_PATH):
        print(f"Local raw dataset already exists at {RAW_DATA_PATH}. Loading...")
        df = pd.read_csv(RAW_DATA_PATH)
        return df

    print("Fetching dataset ID 697 from UCI Machine Learning Repository (ucimlrepo)...")
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=697)
        X = dataset.data.features
        y = dataset.data.targets
        
        # Combine into single DataFrame for raw storage
        df = pd.concat([X, y], axis=1)
        
        # Save to data/raw/dataset.csv
        df.to_csv(RAW_DATA_PATH, index=False)
        print(f"Successfully downloaded and saved {df.shape[0]} rows and {df.shape[1]} columns to {RAW_DATA_PATH}")
        return df
    except Exception as e:
        print(f"Error fetching from ucimlrepo: {e}")
        print("\nFALLBACK INSTRUCTIONS:")
        print("Please place the official UCI 'Predict Students' Dropout and Academic Success' CSV file at:")
        print(f"  {os.path.abspath(RAW_DATA_PATH)}")
        print("The CSV must contain the original 36 features and the 'Target' column.\n")
        raise

def inspect_dataset(df: pd.DataFrame):
    print("=" * 60)
    print("DATASET INSPECTION SUMMARY")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Missing values total: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("\nClass Distribution:")
    target_col = "Target"
    class_counts = df[target_col].value_counts()
    class_props = df[target_col].value_counts(normalize=True) * 100
    for cls in class_counts.index:
        print(f"  - {cls}: {class_counts[cls]} ({class_props[cls]:.2f}%)")
        
    print("\nColumn Data Types:")
    print(df.dtypes.value_counts())
    
    print("\nDetailed Column Stats Summary written to console.")
    print("=" * 60)

if __name__ == "__main__":
    df = fetch_and_save_data()
    inspect_dataset(df)
