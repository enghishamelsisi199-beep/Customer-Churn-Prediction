# Customer Churn Prediction

End-to-end churn prediction on the Telco Customer Churn dataset: data preparation, model training and tuning, a FastAPI service for predictions/retraining, and a Streamlit dashboard.

```
customer-churn-prediction/
├── data_pipeline/    # Step 1 - clean the raw CSV, EDA, train/test split
├── model_training/   # Step 2 - train, evaluate and tune the model (saved to models/)
├── api/              # Step 3 - FastAPI service (predict + retrain endpoints)
└── dashboard/        # Step 4 - Streamlit dashboard
```

See `PROJECT_EXPLANATION.md` for the full walkthrough (problem, data dictionary, method) and `Customer_Churn_Prediction.pptx` for the presentation.

## Run locally

Each folder has its own `requirements.txt`, `.env.example` and README. In each folder, install, copy the env file, then run:

```bash
pip install -r requirements.txt
cp .env.example .env
```

| Step | Folder | Command |
|------|--------|---------|
| 1 | `data_pipeline` | `python prepare_data.py` |
| 2 | `model_training` | `python train.py` then `python evaluate.py` |
| 3 | `api` | `uvicorn app.main:app --reload --port 8001` |
| 4 | `dashboard` | `streamlit run app.py` |

Trained models and the dataset are included, so you can start from step 3 or 4.
