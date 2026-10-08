# Dashboard Step — Churn Prediction

Step 3 (final) of the Customer Churn Prediction project: an interactive
Streamlit dashboard with What-if analysis, batch prediction, cost-sensitive
thresholding, automated retention alerts, and model performance + SHAP
explainability.

## Setup

```bash
cd dashboard
pip install -r requirements.txt
cp .env.example .env
```

Make sure the **Model Training** step ran first (`MODEL_PATH` in `.env`
points to the saved pipeline — the tuned one if you ran `tune_models.py`,
otherwise the plain `churn_model.pkl`). `SAMPLE_DATA_PATH` should point to
the test set from the **Data** step — it's used both to build the What-if
form's input options and as the held-out set on the Performance tab.

## Run

```bash
streamlit run app.py
```

## The five tabs

### 🔮 What-If Analysis
Pick values for a single customer (form built automatically from the real
data's columns and categories) and get:
- Churn probability + a Low/Medium/High risk label
- The top 5 SHAP factors driving that specific prediction
- A detailed SHAP waterfall plot
- If risk is Medium/High: a suggested retention email (subject + body),
  picked by simple, explainable rules (contract type, tenure, monthly
  spend, current add-ons) — not another black-box model

You only fill in **raw** fields (tenure, Contract, OnlineSecurity, etc.) —
the derived features from the Data step (`tenure_group`,
`num_addon_services`, `avg_monthly_spend`) are computed automatically
behind the scenes, so they can never get out of sync with what you entered.

### 📂 Batch Prediction
Upload a CSV of many customers (same raw columns), get a probability +
Yes/No prediction for each, sorted by risk, with a CSV download. Below
that, an **Automated Retention Alerts** section generates a ready-to-send
retention message for every customer at or above a chosen risk cutoff,
using the same rule-based offer logic as the What-If tab — download all
of them as a CSV.

### 💰 Cost-Sensitive Threshold
A fixed 0.5 cutoff treats a missed churner and a wasted retention offer as
equally costly — in reality they rarely are. Enter the real cost of each
(e.g. $500 lost revenue per missed churner vs $50 per unnecessary retention
offer), and this tab finds the threshold that minimizes **total cost** on
the test set — not just the best F1-score. Shows a cost-vs-threshold chart
and a breakdown of false negatives/positives at the optimal point. Hit
**Apply** to make that threshold the one used by the What-If and Batch
Prediction tabs for the rest of the session.

### 📡 Model Monitoring
A model's real-world accuracy silently degrades when the customers it
sees start to differ from what it was trained on ("data drift"). Every
time you run a batch prediction, this tab automatically compares that
batch against the test set using the **Population Stability Index (PSI)**
for each key feature (tenure, MonthlyCharges, Contract, InternetService,
etc.), flags any that show significant drift, and logs the batch's
summary stats (mean predicted risk, rows scored, drift flags) to a local
file so you can see the trend across multiple sessions over time.

### 📊 Model Performance
F1-score, AUC-PR, full classification report, and confusion matrix on the
test set (using whatever threshold is currently active) — plus a SHAP
beeswarm plot showing **global** feature importance (which features matter
most across the whole test set, not just one customer). If
`tuning_report.json` exists (from `tune_models.py`), the model comparison
table is shown here too.

## How the SHAP integration works (`core/shap_utils.py`)

The saved pipeline is `[preprocess -> smote -> model]`. SHAP needs the
already-encoded numeric features and the raw classifier — so
`transform_features()` runs just the `preprocess` step (SMOTE is a no-op at
inference time anyway) and keeps the proper one-hot column names.

The explainer picks its algorithm based on which model won training:
- **Tree models** (XGBoost, Random Forest) → `TreeExplainer` with
  `tree_path_dependent` perturbation — reads the tree structure directly,
  no background dataset needed.
- **Everything else** (e.g. Logistic Regression) → the generic
  `shap.Explainer`, given a background sample from the test set.

This means the dashboard works regardless of which model
`train.py`/`tune_models.py` ended up picking as best.
