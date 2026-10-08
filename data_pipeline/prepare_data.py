"""
Full data preparation pipeline for the Churn Prediction project:
raw CSV -> cleaned -> feature-engineered -> stratified train/test split -> saved.

Usage:
    python prepare_data.py
"""

import os

from sklearn.model_selection import train_test_split

from config import settings
from cleaning import load_raw_data, clean_data
from feature_engineering import engineer_features


def main():
    print("=== Step 1/4: Load raw data ===")
    df = load_raw_data()

    print("\n=== Step 2/4: Clean data ===")
    df = clean_data(df)

    print("\n=== Step 3/4: Feature engineering ===")
    df = engineer_features(df)

    print("\n=== Step 4/4: Train/test split ===")
    train_df, test_df = train_test_split(
        df,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
        stratify=df["Churn"],  # keep the same churn rate in both splits
    )

    os.makedirs(settings.PROCESSED_DATA_DIR, exist_ok=True)
    train_path = os.path.join(settings.PROCESSED_DATA_DIR, "train.csv")
    test_path = os.path.join(settings.PROCESSED_DATA_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nTrain set: {train_df.shape[0]} rows (churn rate: {train_df['Churn'].mean():.2%})")
    print(f"Test set:  {test_df.shape[0]} rows (churn rate: {test_df['Churn'].mean():.2%})")
    print(f"\nSaved to:\n  {train_path}\n  {test_path}")


if __name__ == "__main__":
    main()
