"""
Cleans the raw Telco Customer Churn CSV (the well-known Kaggle/IBM dataset).

Known quirks in this dataset that this module handles:
- `TotalCharges` is stored as a string and has blank values for customers
  with tenure == 0 (brand new customers who haven't been charged yet).
- `customerID` is a unique identifier with no predictive value.
- The target `Churn` is "Yes"/"No" text, not 0/1.
"""

import pandas as pd

from config import settings


def load_raw_data(path: str = None) -> pd.DataFrame:
    path = path or settings.RAW_DATA_PATH
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges: blank strings -> NaN -> numeric. Blanks only occur when
    # tenure == 0 (customer just signed up), so 0 is the correct fill value.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isna().sum()
    if n_missing:
        print(f"Filling {n_missing} missing TotalCharges values with 0 "
              f"(these are all tenure == 0 customers)")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Drop the identifier column — has no predictive signal.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Encode the target as 0/1.
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop exact duplicate rows, if any.
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        print(f"Dropped {before - len(df)} duplicate rows")

    print(f"Cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Churn rate: {df['Churn'].mean():.2%}")

    return df
