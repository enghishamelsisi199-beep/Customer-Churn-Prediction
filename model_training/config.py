import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    TRAIN_PATH: str = os.getenv("TRAIN_PATH", "../data_pipeline/data/processed/train.csv")
    TEST_PATH: str = os.getenv("TEST_PATH", "../data_pipeline/data/processed/test.csv")
    MODEL_OUTPUT_DIR: str = os.getenv("MODEL_OUTPUT_DIR", "./models")
    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))


settings = Settings()
