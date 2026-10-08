"""
Detects whether newly uploaded customer data looks statistically different
from the data the model was trained/tested on — a signal that predictions
may be getting less reliable and a retrain might be worth considering.

Uses the Population Stability Index (PSI), a standard, easy-to-explain
drift metric:
    PSI < 0.10           -> stable, no real shift
    0.10 <= PSI < 0.25    -> moderate shift, worth watching
    PSI >= 0.25           -> significant shift, investigate / consider retraining
"""

import numpy as np
import pandas as pd

PSI_MODERATE_THRESHOLD = 0.10
PSI_SIGNIFICANT_THRESHOLD = 0.25

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "Contract", "InternetService", "PaymentMethod", "PaperlessBilling",
    "OnlineSecurity", "TechSupport", "SeniorCitizen",
]


def _psi_from_proportions(ref_props: np.ndarray, cur_props: np.ndarray) -> float:
    # Avoid log(0) / division by zero on empty bins/categories
    eps = 1e-4
    ref_props = np.where(ref_props == 0, eps, ref_props)
    cur_props = np.where(cur_props == 0, eps, cur_props)
    return float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))


def compute_psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = reference.dropna()
    current = current.dropna()

    # Bin edges from the REFERENCE distribution only — current data is
    # measured against those same bins, however it falls into them.
    try:
        bin_edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            return 0.0  # not enough variation to bin meaningfully
    except Exception:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_props = ref_counts / max(ref_counts.sum(), 1)
    cur_props = cur_counts / max(cur_counts.sum(), 1)

    return _psi_from_proportions(ref_props, cur_props)


def compute_psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    categories = sorted(set(reference.dropna().unique()) | set(current.dropna().unique()))

    ref_counts = reference.value_counts().reindex(categories, fill_value=0)
    cur_counts = current.value_counts().reindex(categories, fill_value=0)

    ref_props = (ref_counts / max(ref_counts.sum(), 1)).values
    cur_props = (cur_counts / max(cur_counts.sum(), 1)).values

    return _psi_from_proportions(ref_props, cur_props)


def _status_for_psi(psi: float) -> str:
    if psi >= PSI_SIGNIFICANT_THRESHOLD:
        return "🔴 Significant drift"
    if psi >= PSI_MODERATE_THRESHOLD:
        return "🟡 Moderate drift"
    return "🟢 Stable"


def compute_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compares `current_df` (e.g. a newly uploaded batch) against
    `reference_df` (e.g. the test set the model was evaluated on) for a
    fixed set of features known to matter for this dataset. Returns one
    row per feature with its PSI score and drift status.
    """
    rows = []

    for feature in NUMERIC_FEATURES:
        if feature in reference_df.columns and feature in current_df.columns:
            psi = compute_psi_numeric(reference_df[feature], current_df[feature])
            rows.append({"feature": feature, "type": "numeric", "psi": round(psi, 4),
                         "status": _status_for_psi(psi)})

    for feature in CATEGORICAL_FEATURES:
        if feature in reference_df.columns and feature in current_df.columns:
            psi = compute_psi_categorical(
                reference_df[feature].astype(str), current_df[feature].astype(str)
            )
            rows.append({"feature": feature, "type": "categorical", "psi": round(psi, 4),
                         "status": _status_for_psi(psi)})

    report = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return report
