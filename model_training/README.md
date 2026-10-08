# Model Training Step — Churn Prediction

Step 2 of the Customer Churn Prediction project: trains and compares three
models with SMOTE for class imbalance, and saves the best one.

## Setup

```bash
cd model_training
pip install -r requirements.txt
cp .env.example .env
```

Make sure the **Data step** ran first — `TRAIN_PATH`/`TEST_PATH` in `.env`
point to `../data_pipeline/data/processed/{train,test}.csv` by default.

## Run

```bash
python train.py
```

This will:
1. Load `train.csv` / `test.csv`
2. Build a preprocessing step (StandardScaler for numeric columns,
   OneHotEncoder for categorical columns) — fit only on training data
3. For each of **Logistic Regression**, **Random Forest**, and **XGBoost**:
   - Wrap it in a pipeline: `preprocessing -> SMOTE -> model`
     (SMOTE only oversamples the minority class during *training* — it's
     never applied to the test set, thanks to `imblearn`'s Pipeline)
   - Fit on the training set, evaluate on the held-out test set
   - Score with **F1-score** and **AUC-PR** (better than plain accuracy on
     imbalanced data like this ~26% churn rate)
4. Pick the best model by F1-score and save the **full pipeline**
   (preprocessing + SMOTE + model) as one file — so at inference time you
   just load one `.pkl` and call `.predict()` on raw-ish data, no separate
   encoding step needed

## Re-check results without re-training

```bash
python evaluate.py
```

Prints F1-score, AUC-PR, a full classification report, and a confusion
matrix for the saved model.

## Output of this step

- `models/churn_model.pkl` — the full trained pipeline (preprocessing +
  SMOTE + best model), ready to be loaded by the Streamlit dashboard
- `models/metrics_report.json` — F1/AUC-PR for all 3 models compared, plus
  which one won

## Notes

- SMOTE is applied **after** preprocessing (needs numeric data) and
  **only** to the training fold — `imblearn.pipeline.Pipeline` (not
  scikit-learn's plain `Pipeline`) handles this correctly, skipping SMOTE
  automatically at predict time.
- `RandomForestClassifier(class_weight=None)` — this is intentional: SMOTE
  already handles the imbalance, so an additional `class_weight` would
  double-compensate. Feel free to experiment with `class_weight="balanced"`
  and no SMOTE as an alternative to compare.
