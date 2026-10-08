from typing import Literal
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """Raw customer fields — same schema as the original Telco dataset,
    minus customerID (no predictive value) and the derived features
    (tenure_group, num_addon_services, avg_monthly_spend), which the API
    computes automatically."""

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 845.50,
            }
        }
    }


class PredictionResponse(BaseModel):
    churn_probability: float
    predicted_churn: Literal["Yes", "No"]
    risk_level: Literal["Low", "Medium", "High"]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str


class RetrainResponse(BaseModel):
    rows_used: int
    candidate_results: dict
    best_candidate: str
    current_model_metrics_on_new_data: dict
    replaced_model: bool
    backed_up_previous_model_to: str | None = None
    new_model_saved_to: str | None = None
