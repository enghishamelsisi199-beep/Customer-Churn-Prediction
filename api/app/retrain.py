"""
Core logic for POST /retrain:

1. Clean + feature-engineer the newly uploaded, labeled CSV.
2. Split it into train/test.
3. Train a couple of candidate pipelines (Logistic Regression, XGBoost)
   with SMOTE, same approach as model_training/train.py.
4. Evaluate every candidate AND the currently-loaded model on the SAME new
   test split, so the comparison is apples-to-apples.
5. If the best new candidate beats the current model's F1-score, back up
   the old model file, save the new one, and hot-swap it into memory.
   Otherwise, keep the current model and explain why.
"""

import os
import shutil
from datetime import datetime

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score

from app.config import settings
from app.cleaning import clean_data
from app.feature_engineering import engineer_features
from app.preprocessing import build_preprocessor, split_features_target


def _build_candidates(preprocessor, random_state: int):
    return {
        "Logistic Regression": ImbPipeline([
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=random_state)),
            ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]),
        "XGBoost": ImbPipeline([
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=random_state)),
            ("model", XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                eval_metric="logloss", random_state=random_state,
            )),
        ]),
    }


def _evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "auc_pr": round(float(average_precision_score(y_test, y_proba)), 4),
    }


def run_retrain(raw_df: pd.DataFrame, current_pipeline):
    """Returns a report dict; also hot-swaps + persists the new model if it wins."""

    if len(raw_df) < settings.MIN_ROWS_FOR_RETRAIN:
        raise ValueError(
            f"Uploaded file has only {len(raw_df)} rows — need at least "
            f"{settings.MIN_ROWS_FOR_RETRAIN} to retrain reliably."
        )
    if "Churn" not in raw_df.columns:
        raise ValueError("Uploaded file must include a 'Churn' (Yes/No) column to retrain on.")

    df = clean_data(raw_df)
    df = engineer_features(df)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=settings.RANDOM_STATE, stratify=df["Churn"],
    )
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    preprocessor = build_preprocessor(train_df)
    candidates = _build_candidates(preprocessor, settings.RANDOM_STATE)

    candidate_results = {}
    fitted = {}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        candidate_results[name] = _evaluate(pipeline, X_test, y_test)
        fitted[name] = pipeline

    best_name = max(candidate_results, key=lambda n: candidate_results[n]["f1_score"])
    best_new_pipeline = fitted[best_name]
    best_new_metrics = candidate_results[best_name]

    # Evaluate the CURRENTLY LOADED model on this exact same new test split,
    # so "better" is measured on the same data, fairly.
    try:
        current_metrics = _evaluate(current_pipeline, X_test, y_test)
    except Exception as e:
        # Current model can't even score this new data's schema — treat as
        # automatically beaten, since it can no longer be trusted on new data.
        current_metrics = {"f1_score": -1.0, "auc_pr": -1.0, "error": str(e)}

    replaced = best_new_metrics["f1_score"] > current_metrics["f1_score"]

    report = {
        "rows_used": len(df),
        "candidate_results": candidate_results,
        "best_candidate": best_name,
        "current_model_metrics_on_new_data": current_metrics,
        "replaced_model": replaced,
    }

    if replaced:
        os.makedirs(settings.MODEL_BACKUP_DIR, exist_ok=True)
        if os.path.exists(settings.MODEL_PATH):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(
                settings.MODEL_BACKUP_DIR, f"churn_model_{timestamp}.pkl"
            )
            shutil.copy(settings.MODEL_PATH, backup_path)
            report["backed_up_previous_model_to"] = backup_path

        joblib.dump(best_new_pipeline, settings.MODEL_PATH)
        report["new_model_saved_to"] = settings.MODEL_PATH

    return report, (best_new_pipeline if replaced else current_pipeline)
