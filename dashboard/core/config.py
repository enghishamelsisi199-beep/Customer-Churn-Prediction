import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    MODEL_PATH: str = os.getenv("MODEL_PATH", "../model_training/models/churn_model_tuned.pkl")
    SAMPLE_DATA_PATH: str = os.getenv("SAMPLE_DATA_PATH", "../data_pipeline/data/processed/test.csv")
    CHURN_THRESHOLD: float = float(os.getenv("CHURN_THRESHOLD", "0.5"))


settings = Settings()
