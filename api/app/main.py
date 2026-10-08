import io

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import settings
from app.model_loader import get_pipeline, set_pipeline
from app.feature_engineering import engineer_features
from app.retrain import run_retrain
from app.schemas import CustomerInput, PredictionResponse, HealthResponse, RetrainResponse

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability from raw Telco-style customer data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _risk_level(probability: float) -> str:
    if probability >= 0.6:
        return "High"
    if probability >= 0.35:
        return "Medium"
    return "Low"


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=get_pipeline() is not None,
        model_path=settings.MODEL_PATH,
    )


@app.get("/schema")
def get_input_schema():
    """Returns the JSON schema for CustomerInput — useful for any frontend
    (Streamlit, React, etc.) that wants to build an input form dynamically
    without hardcoding the field list."""
    return CustomerInput.model_json_schema()


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    raw_df = pd.DataFrame([customer.model_dump()])
    engineered_df = engineer_features(raw_df)
    pipeline = get_pipeline()

    try:
        probability = float(pipeline.predict_proba(engineered_df)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    predicted = "Yes" if probability >= settings.CHURN_THRESHOLD else "No"

    return PredictionResponse(
        churn_probability=round(probability, 4),
        predicted_churn=predicted,
        risk_level=_risk_level(probability),
    )


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Accepts a CSV with the same raw columns as CustomerInput (no customerID/
    Churn/derived columns needed) and returns a CSV with two extra columns:
    churn_probability and predicted_churn.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    try:
        engineered_df = engineer_features(df)
        pipeline = get_pipeline()
        probabilities = pipeline.predict_proba(engineered_df)[:, 1]
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Prediction failed — check the CSV has all required columns: {e}",
        )

    result_df = df.copy()
    result_df["churn_probability"] = probabilities
    result_df["predicted_churn"] = (probabilities >= settings.CHURN_THRESHOLD)
    result_df["predicted_churn"] = result_df["predicted_churn"].map({True: "Yes", False: "No"})

    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return StreamingResponse(
        iter([csv_buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=churn_predictions.csv"},
    )


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(file: UploadFile = File(...)):
    """
    Upload a fresh, LABELED CSV (raw Telco-style columns + a 'Churn' Yes/No
    column) to retrain candidate models on it. If the best new candidate
    beats the currently-loaded model's F1-score on a held-out split of this
    same new data, the model is hot-swapped and persisted to disk (with the
    previous version backed up first). Otherwise, the current model is kept
    and the report explains why.
    """
    try:
        contents = await file.read()
        raw_df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    try:
        report, active_pipeline = run_retrain(raw_df, get_pipeline())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e}")

    if report["replaced_model"]:
        set_pipeline(active_pipeline)

    return RetrainResponse(**report)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
