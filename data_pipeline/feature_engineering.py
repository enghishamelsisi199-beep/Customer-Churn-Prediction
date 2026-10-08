"""
Adds a few derived features on top of the cleaned Telco Churn data.
Categorical columns are intentionally left as strings here (not one-hot
encoded) — encoding + scaling happens later, inside the model-training
pipeline, so it's fit only on the training split and never leaks into test.
"""

import pandas as pd

ADD_ON_SERVICE_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Tenure buckets — churn behavior often differs a lot by customer "age".
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 72],
        labels=["0-12mo", "13-24mo", "25-48mo", "49-60mo", "61-72mo"],
    ).astype(str)

    # Count of add-on services actively subscribed (treat "No"/"No internet
    # service" both as "not subscribed").
    def count_services(row):
        return sum(1 for col in ADD_ON_SERVICE_COLUMNS if row[col] == "Yes")

    df["num_addon_services"] = df.apply(count_services, axis=1)

    # Average monthly spend over the customer's lifetime so far — captures
    # customers whose current MonthlyCharges differs a lot from their history
    # (e.g. recently upgraded/downgraded).
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    print(f"Added features: tenure_group, num_addon_services, avg_monthly_spend")
    return df
