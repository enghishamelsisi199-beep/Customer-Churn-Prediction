"""
Quick, terminal-only EDA on the cleaned+engineered data — no plots, just
numbers, useful for a fast sanity check before moving to model training.

Usage:
    python eda.py
"""

import pandas as pd

from config import settings
from cleaning import load_raw_data, clean_data
from feature_engineering import engineer_features


def main():
    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)

    print("\n=== Churn rate by contract type ===")
    print(df.groupby("Contract")["Churn"].mean().sort_values(ascending=False).apply(lambda x: f"{x:.1%}"))

    print("\n=== Churn rate by tenure group ===")
    print(df.groupby("tenure_group")["Churn"].mean().apply(lambda x: f"{x:.1%}"))

    print("\n=== Churn rate by number of add-on services ===")
    print(df.groupby("num_addon_services")["Churn"].mean().apply(lambda x: f"{x:.1%}"))

    print("\n=== Churn rate by internet service ===")
    print(df.groupby("InternetService")["Churn"].mean().sort_values(ascending=False).apply(lambda x: f"{x:.1%}"))

    print("\n=== Monthly charges: churned vs not ===")
    print(df.groupby("Churn")["MonthlyCharges"].describe()[["mean", "50%", "std"]])


if __name__ == "__main__":
    main()
