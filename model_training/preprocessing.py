"""
Builds the preprocessing step of the pipeline: one-hot encodes categorical
columns, scales numeric columns. This is fit ONLY on the training data
(inside the pipeline), so there's no leakage from the test set.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Churn"

NUMERIC_COLUMNS = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "num_addon_services", "avg_monthly_spend",
]


def get_categorical_columns(df: pd.DataFrame) -> list:
    return [
        col for col in df.columns
        if col not in NUMERIC_COLUMNS + [TARGET_COLUMN]
    ]


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = get_categorical_columns(df)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y
