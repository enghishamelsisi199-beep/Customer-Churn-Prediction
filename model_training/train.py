"""
Trains and compares Logistic Regression, Random Forest, and XGBoost for
churn prediction, handling class imbalance with SMOTE, and saves the
best-performing full pipeline (preprocessing + SMOTE + model) to disk.

Usage:
    python train.py
"""

import os
import json

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, average_precision_score, classification_report

from config import settings
from preprocessing import build_preprocessor, split_features_target


def build_candidates(preprocessor, random_state: int):
    """Returns {name: full_pipeline} for each model to compare."""
    return {
        "Logistic Regression": ImbPipeline([
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=random_state)),
            ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]),
        "Random Forest": ImbPipeline([
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=random_state)),
            ("model", RandomForestClassifier(
                n_estimators=300, random_state=random_state, class_weight=None
            )),
        ]),
        "XGBoost": ImbPipeline([
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=random_state)),
            ("model", XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.1,
                eval_metric="logloss", random_state=random_state,
            )),
        ]),
    }


def evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "f1_score": f1_score(y_test, y_pred),
        "auc_pr": average_precision_score(y_test, y_proba),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }


def main():
    print("=== Loading data ===")
    train_df = pd.read_csv(settings.TRAIN_PATH)
    test_df = pd.read_csv(settings.TEST_PATH)
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    preprocessor = build_preprocessor(train_df)
    candidates = build_candidates(preprocessor, settings.RANDOM_STATE)

    results = {}
    fitted_pipelines = {}

    for name, pipeline in candidates.items():
        print(f"\n=== Training: {name} ===")
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        results[name] = metrics
        fitted_pipelines[name] = pipeline
        print(f"F1-score: {metrics['f1_score']:.4f} | AUC-PR: {metrics['auc_pr']:.4f}")

    print("\n=== Model comparison (sorted by F1-score) ===")
    ranked = sorted(results.items(), key=lambda kv: kv[1]["f1_score"], reverse=True)
    for name, metrics in ranked:
        print(f"{name:<25} F1: {metrics['f1_score']:.4f}  AUC-PR: {metrics['auc_pr']:.4f}")

    best_name, best_metrics = ranked[0]
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name}")

    os.makedirs(settings.MODEL_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(settings.MODEL_OUTPUT_DIR, "churn_model.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"Saved best pipeline (preprocessing + model) to: {model_path}")

    report_path = os.path.join(settings.MODEL_OUTPUT_DIR, "metrics_report.json")
    with open(report_path, "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "all_results": {
                    name: {"f1_score": m["f1_score"], "auc_pr": m["auc_pr"]}
                    for name, m in results.items()
                },
            },
            f,
            indent=2,
        )
    print(f"Saved metrics report to: {report_path}")


if __name__ == "__main__":
    main()
