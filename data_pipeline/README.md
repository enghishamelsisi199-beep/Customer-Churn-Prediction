# Data Step — Churn Prediction

Step 1 of the Customer Churn Prediction project: download the dataset,
clean it, engineer a few features, and produce a stratified train/test split.

## Get the dataset

This project uses the well-known **Telco Customer Churn** dataset (IBM /
Kaggle). Download it manually (free, no API key needed):

1. Go to [kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Sign in (free account) → click **Download**
3. Unzip and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` into
   `data/raw/`, renamed to `Telco-Customer-Churn.csv`
   (or update `RAW_DATA_PATH` in `.env` to match whatever name you kept)

## Setup

```bash
cd data_pipeline
pip install -r requirements.txt
cp .env.example .env
```

## Run

```bash
python prepare_data.py
```

This will:
1. Load the raw CSV
2. Clean it (fix `TotalCharges`, drop the ID column, encode the target)
3. Engineer 3 extra features: `tenure_group`, `num_addon_services`,
   `avg_monthly_spend`
4. Split into a stratified 80/20 train/test set (same churn rate in both)
5. Save `data/processed/train.csv` and `data/processed/test.csv`

Optional — a quick terminal EDA on churn patterns (no plots):
```bash
python eda.py
```

## Output of this step

- `data/processed/train.csv` — for model training (next step)
- `data/processed/test.csv` — held out, only used for final evaluation

Categorical columns are **intentionally left un-encoded** here (e.g.
`Contract`, `InternetService` are still text). Encoding + scaling will be
built as part of the model-training pipeline next, fit only on the training
split, so there's no data leakage into the test set.

## Notes on the data

- ~26% of customers in this dataset churned (moderately imbalanced) — the
  model-training step will need to account for this (SMOTE / class weights).
- `MonthlyCharges` and `Contract` type are historically the strongest churn
  signals in this dataset — worth checking those first in `eda.py`'s output.
