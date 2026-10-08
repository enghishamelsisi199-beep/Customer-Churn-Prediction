"""
Same cleaning logic as data_pipeline/cleaning.py, duplicated here so the
API is self-contained and can clean freshly-uploaded data for retraining
without depending on the data_pipeline/ folder.
"""

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if not pd.api.types.is_numeric_dtype(df["Churn"]):
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    df = df.drop_duplicates()

    return df
