"""
Loads the saved best model and prints a full evaluation on the test set —
useful to re-check results without re-training.

Usage:
    python evaluate.py
"""

import os

import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, average_precision_score,
)

from config import settings
from preprocessing import split_features_target


def main():
    model_path = os.path.join(settings.MODEL_OUTPUT_DIR, "churn_model.pkl")
    pipeline = joblib.load(model_path)
    print(f"Loaded model from: {model_path}")

    test_df = pd.read_csv(settings.TEST_PATH)
    X_test, y_test = split_features_target(test_df)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(f"\nF1-score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-PR:   {average_precision_score(y_test, y_proba):.4f}")

    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    print("=== Confusion matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                Predicted No   Predicted Yes")
    print(f"Actual No       {cm[0][0]:<14} {cm[0][1]}")
    print(f"Actual Yes      {cm[1][0]:<14} {cm[1][1]}")


if __name__ == "__main__":
    main()
