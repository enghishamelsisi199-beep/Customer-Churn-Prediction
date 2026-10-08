"""
A lightweight, file-based monitoring log: every time a batch prediction
runs, a summary row (timestamp, batch size, mean predicted churn
probability, drift flags) is appended to a local JSONL file. This lets the
Model Monitoring tab show a trend over multiple sessions/days, without
needing a real database for what's a portfolio-scale project.
"""

import json
import os
from datetime import datetime

import pandas as pd

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "monitoring_log.jsonl")


def log_batch(n_rows: int, mean_probability: float, n_flagged_high_risk: int,
              max_psi: float = None, drifted_features: int = 0) -> None:
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_rows": n_rows,
        "mean_churn_probability": round(float(mean_probability), 4),
        "n_flagged_high_risk": n_flagged_high_risk,
        "max_psi": round(float(max_psi), 4) if max_psi is not None else None,
        "drifted_features": drifted_features,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_log() -> pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=[
            "timestamp", "n_rows", "mean_churn_probability",
            "n_flagged_high_risk", "max_psi", "drifted_features",
        ])

    rows = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
