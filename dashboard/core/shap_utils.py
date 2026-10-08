"""
SHAP explainability for the churn pipeline. The saved pipeline is
[preprocess -> smote -> model]; SHAP needs the already-preprocessed
(numeric, one-hot-encoded) features and the raw classifier, not the
whole pipeline (SMOTE is a no-op at inference time anyway).
"""

import numpy as np
import pandas as pd
import shap
import streamlit as st
from scipy.sparse import issparse


def _to_dense(X):
    return X.toarray() if issparse(X) else X


def transform_features(pipeline, X_df: pd.DataFrame):
    """Runs just the preprocessing step, returning a dense DataFrame with
    proper (one-hot-encoded) column names — what SHAP needs."""
    preprocessor = pipeline.named_steps["preprocess"]
    X_transformed = _to_dense(preprocessor.transform(X_df))
    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(X_transformed, columns=feature_names, index=X_df.index)


@st.cache_resource(show_spinner=False)
def get_explainer(_pipeline, _background_df: pd.DataFrame):
    """
    Cached — building the explainer is the expensive part, not using it.
    Leading underscores on the args tell Streamlit not to try to hash them.

    Picks the right SHAP algorithm for whatever model won during training:
    - Tree-based models (XGBoost, Random Forest) use TreeExplainer with
      tree_path_dependent perturbation, which reads the split structure
      directly and needs no background dataset. Passing a background
      dataset here trips an unrelated XGBoost/SHAP compatibility bug on
      certain versions, so we deliberately don't for tree models.
    - Everything else (e.g. Logistic Regression) uses the generic
      shap.Explainer, which needs a background sample.
    """
    model = _pipeline.named_steps["model"]
    model_type_name = type(model).__name__

    if model_type_name in {"XGBClassifier", "RandomForestClassifier"}:
        return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

    background = transform_features(_pipeline, _background_df)
    return shap.Explainer(model, background)


def explain_instance(pipeline, X_row_df: pd.DataFrame, background_df: pd.DataFrame):
    """
    X_row_df: a single-row DataFrame in the ORIGINAL (raw) feature format.
    background_df: a sample of raw rows (e.g. from the test set) used as
    the SHAP reference distribution.
    Returns (shap_explanation_for_row, transformed_row_df) for plotting.
    """
    explainer = get_explainer(pipeline, background_df)
    X_transformed = transform_features(pipeline, X_row_df)
    shap_values = explainer(X_transformed)

    # Binary classifiers sometimes return shape (n, features, 2) — take the
    # "positive class" (churn=1) slice if so.
    if shap_values.values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    return shap_values[0], X_transformed


def explain_batch(pipeline, X_df: pd.DataFrame, background_df: pd.DataFrame):
    """Same as explain_instance but for many rows at once — used for the
    global feature-importance summary plot."""
    explainer = get_explainer(pipeline, background_df)
    X_transformed = transform_features(pipeline, X_df)
    shap_values = explainer(X_transformed)

    if shap_values.values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    return shap_values, X_transformed


def top_contributing_features(shap_explanation, top_n: int = 5) -> pd.DataFrame:
    """Returns a small table of the top_n features pushing this prediction
    toward/away from churn, for a simpler alternative to the plot."""
    values = shap_explanation.values
    names = shap_explanation.feature_names
    data = shap_explanation.data

    df = pd.DataFrame({
        "feature": names,
        "value": data,
        "shap_impact": values,
    })
    df["abs_impact"] = df["shap_impact"].abs()
    df = df.sort_values("abs_impact", ascending=False).head(top_n)
    df["direction"] = df["shap_impact"].apply(lambda v: "increases churn risk" if v > 0 else "decreases churn risk")
    return df[["feature", "value", "shap_impact", "direction"]]
