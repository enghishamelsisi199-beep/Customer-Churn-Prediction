"""
Same derived features as data_pipeline/feature_engineering.py, duplicated
here so the API is self-contained. Callers send RAW fields only — these
derived features are computed automatically before prediction.
"""

import pandas as pd

ADD_ON_SERVICE_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

DERIVED_COLUMNS = ["tenure_group", "num_addon_services", "avg_monthly_spend"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 72],
        labels=["0-12mo", "13-24mo", "25-48mo", "49-60mo", "61-72mo"],
    ).astype(str)

    def count_services(row):
        return sum(1 for col in ADD_ON_SERVICE_COLUMNS if row[col] == "Yes")

    df["num_addon_services"] = df.apply(count_services, axis=1)
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    return df
