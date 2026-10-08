"""
Hyperparameter tuning for the churn models, optimizing for F1-score (not
accuracy — with a ~26% churn rate, accuracy rewards a model that just
predicts "No Churn" every time). Uses RandomizedSearchCV with stratified
cross-validation on the training set only; the test set stays untouched
until the very end.

Usage:
    python tune_models.py
"""

import os
import json

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score, classification_report

from config import settings
from preprocessing import build_preprocessor, split_features_target


def tune_logistic_regression(preprocessor, X_train, y_train, random_state):
    pipeline = ImbPipeline([
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=random_state)),
        ("model", LogisticRegression(max_iter=2000, random_state=random_state)),
    ])

    param_distributions = {
        "model__C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
        "model__class_weight": [None, "balanced"],
        "smote__k_neighbors": [3, 5, 7],
    }

    return _search(pipeline, param_distributions, X_train, y_train, random_state, n_iter=20)


def tune_xgboost(preprocessor, X_train, y_train, random_state):
    pipeline = ImbPipeline([
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=random_state)),
        ("model", XGBClassifier(eval_metric="logloss", random_state=random_state)),
    ])

    param_distributions = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5],
        "smote__k_neighbors": [3, 5, 7],
    }

    return _search(pipeline, param_distributions, X_train, y_train, random_state, n_iter=30)


def _search(pipeline, param_distributions, X_train, y_train, random_state, n_iter):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"  Best CV F1-score: {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")
    return search.best_estimator_, search.best_score_, search.best_params_


def evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "f1_score": f1_score(y_test, y_pred),
        "auc_pr": average_precision_score(y_test, y_proba),
    }


def main():
    print("=== Loading data ===")
    train_df = pd.read_csv(settings.TRAIN_PATH)
    test_df = pd.read_csv(settings.TEST_PATH)
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    preprocessor = build_preprocessor(train_df)

    results = {}
    fitted = {}

    print("\n=== Tuning: Logistic Regression ===")
    model, cv_f1, params = tune_logistic_regression(preprocessor, X_train, y_train, settings.RANDOM_STATE)
    fitted["Logistic Regression (tuned)"] = model
    results["Logistic Regression (tuned)"] = {"cv_f1": cv_f1, "best_params": params}

    print("\n=== Tuning: XGBoost ===")
    model, cv_f1, params = tune_xgboost(preprocessor, X_train, y_train, settings.RANDOM_STATE)
    fitted["XGBoost (tuned)"] = model
    results["XGBoost (tuned)"] = {"cv_f1": cv_f1, "best_params": params}

    print("\n=== Test set evaluation (final, untouched during tuning) ===")
    for name, pipeline in fitted.items():
        test_metrics = evaluate(pipeline, X_test, y_test)
        results[name].update(test_metrics)
        print(f"{name:<30} Test F1: {test_metrics['f1_score']:.4f}  "
              f"Test AUC-PR: {test_metrics['auc_pr']:.4f}")

    best_name = max(fitted, key=lambda n: results[n]["f1_score"])
    best_pipeline = fitted[best_name]
    print(f"\nBest tuned model: {best_name}")
    print(classification_report(y_test, best_pipeline.predict(X_test), target_names=["No Churn", "Churn"]))

    os.makedirs(settings.MODEL_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(settings.MODEL_OUTPUT_DIR, "churn_model_tuned.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"\nSaved tuned pipeline to: {model_path}")

    report_path = os.path.join(settings.MODEL_OUTPUT_DIR, "tuning_report.json")
    with open(report_path, "w") as f:
        json.dump({"best_model": best_name, "all_results": results}, f, indent=2, default=str)
    print(f"Saved tuning report to: {report_path}")


if __name__ == "__main__":
    main()
