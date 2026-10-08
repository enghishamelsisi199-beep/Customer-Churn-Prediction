import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    MODEL_PATH: str = os.getenv("MODEL_PATH", "../model_training/models/churn_model_tuned.pkl")
    MODEL_BACKUP_DIR: str = os.getenv("MODEL_BACKUP_DIR", "./model_backups")
    CHURN_THRESHOLD: float = float(os.getenv("CHURN_THRESHOLD", "0.5"))
    MIN_ROWS_FOR_RETRAIN: int = int(os.getenv("MIN_ROWS_FOR_RETRAIN", "100"))
    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))
    HOST: str = os.getenv("API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("API_PORT", "8001"))
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "*").split(",")


settings = Settings()
