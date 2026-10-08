"""
Finds the probability threshold that minimizes total expected business
cost, instead of using a fixed 0.5 cutoff.

Two costs matter, and they're rarely equal in practice:
- cost_fn: the cost of a FALSE NEGATIVE — missing a customer who actually
  churns (lost revenue: their full future value).
- cost_fp: the cost of a FALSE POSITIVE — flagging a loyal customer as
  at-risk (cost of an unnecessary retention offer/discount).

For each candidate threshold, predicted "churn" = probability >= threshold.
Total cost = (false negatives * cost_fn) + (false positives * cost_fp).
The optimal threshold is whichever minimizes that total across the test set.
"""

import numpy as np
import pandas as pd


def compute_cost_curve(
    y_true, y_proba, cost_fn: float, cost_fp: float, n_thresholds: int = 101
) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    rows = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        false_negatives = int(((y_pred == 0) & (y_true == 1)).sum())
        false_positives = int(((y_pred == 1) & (y_true == 0)).sum())
        true_positives = int(((y_pred == 1) & (y_true == 1)).sum())
        true_negatives = int(((y_pred == 0) & (y_true == 0)).sum())

        total_cost = false_negatives * cost_fn + false_positives * cost_fp

        rows.append({
            "threshold": round(float(t), 3),
            "total_cost": total_cost,
            "false_negatives": false_negatives,
            "false_positives": false_positives,
            "true_positives": true_positives,
            "true_negatives": true_negatives,
        })

    return pd.DataFrame(rows)


def find_optimal_threshold(cost_curve: pd.DataFrame) -> dict:
    best_row = cost_curve.loc[cost_curve["total_cost"].idxmin()]
    return best_row.to_dict()
