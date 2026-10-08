import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    RAW_DATA_PATH: str = os.getenv("RAW_DATA_PATH", "./data/raw/Telco-Customer-Churn.csv")
    PROCESSED_DATA_DIR: str = os.getenv("PROCESSED_DATA_DIR", "./data/processed")
    TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.2"))
    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))


settings = Settings()
