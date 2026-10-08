# Churn Prediction API

A standalone FastAPI service wrapping the trained churn pipeline — separates
the model from the Streamlit dashboard, so predictions can be called from
anywhere (a CRM, another app, a script), not just the dashboard UI.

## Setup

```bash
cd api
pip install -r requirements.txt
cp .env.example .env
```

Make sure `MODEL_PATH` in `.env` points to a trained pipeline from the
**Model Training** step (tuned or plain).

## Run

```bash
uvicorn app.main:app --reload --port 8001
```

Interactive docs: `http://localhost:8001/docs`

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Check the API and model are up |
| `/schema` | GET | JSON schema of the expected customer input — lets any frontend build a form without hardcoding fields |
| `/predict` | POST | Single customer (raw fields as JSON) → probability + risk level |
| `/predict/batch` | POST | Upload a CSV of customers → get back the same CSV with `churn_probability` and `predicted_churn` columns added |
| `/retrain` | POST | Upload a fresh, **labeled** CSV (raw columns + `Churn`) → trains new candidates, compares them to the current model, hot-swaps if better |

## Example: single prediction

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.35, "TotalCharges": 845.50
  }'
```

Response:
```json
{"churn_probability": 0.62, "predicted_churn": "Yes", "risk_level": "High"}
```

## Notes

- Only **raw** fields are accepted (`CustomerInput` in `app/schemas.py`) —
  the derived features from the Data step (`tenure_group`,
  `num_addon_services`, `avg_monthly_spend`) are computed automatically in
  `app/feature_engineering.py`, so callers never have to compute them or
  risk getting them out of sync.
- Pydantic validates every field (e.g. `Contract` must be one of the three
  real values) — invalid requests get a clear 422 error instead of a
  confusing model failure.
- This is the foundation for the next planned feature: an endpoint that
  accepts a fresh batch of labeled data and retrains the model on it
  (see the project's root README → Future Improvements).

## How `/retrain` works

Upload a CSV with the same raw columns as `/predict/batch`, **plus** a
`Churn` column (`Yes`/`No`) — this needs to be *labeled* data, since it's
used to both train and evaluate new candidates.

1. Cleans + feature-engineers the upload (same logic as the Data step).
2. Splits it 80/20 (stratified) into a fresh train/test set.
3. Trains **Logistic Regression** and **XGBoost** candidates with SMOTE on
   the 80%.
4. Evaluates every candidate, **and the currently-loaded model**, on the
   same 20% test split — so it's an apples-to-apples comparison on data
   none of them has seen.
5. If the best new candidate's F1-score beats the current model's F1-score
   on that same split:
   - the current model file is backed up to `MODEL_BACKUP_DIR` with a
     timestamp (e.g. `churn_model_20260315_142200.pkl`)
   - the new model is saved to `MODEL_PATH`, **and hot-swapped into the
     running API immediately** — no restart needed
   - if not, the current model is kept, and the response explains the
     scores that led to that decision

Response example:
```json
{
  "rows_used": 850,
  "candidate_results": {
    "Logistic Regression": {"f1_score": 0.63, "auc_pr": 0.66},
    "XGBoost": {"f1_score": 0.67, "auc_pr": 0.69}
  },
  "best_candidate": "XGBoost",
  "current_model_metrics_on_new_data": {"f1_score": 0.61, "auc_pr": 0.64},
  "replaced_model": true,
  "backed_up_previous_model_to": "./model_backups/churn_model_20260315_142200.pkl",
  "new_model_saved_to": "../model_training/models/churn_model_tuned.pkl"
}
```

**Rollback:** if a retrain ever makes things worse in practice, copy the
relevant timestamped file from `model_backups/` back over `MODEL_PATH` and
restart the API.

**Note on scope:** this compares 2 quick candidates (no hyperparameter
search) to keep the endpoint's response time reasonable — it's meant for
periodic refreshes on new data, not a replacement for the deeper
`tune_models.py` search when starting from scratch.
