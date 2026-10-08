import joblib
import pandas as pd
import streamlit as st

from core.config import settings
from core.feature_engineering import DERIVED_COLUMNS

TARGET_COLUMN = "Churn"


@st.cache_resource(show_spinner=False)
def load_pipeline():
    return joblib.load(settings.MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    """Used to prefill the What-if form with a realistic existing customer,
    and to know the exact feature schema/categories expected by the model."""
    df = pd.read_csv(settings.SAMPLE_DATA_PATH)
    return df


def get_feature_schema(sample_df: pd.DataFrame) -> dict:
    """
    Returns {column: {"type": "numeric"/"categorical", "options": [...] or (min, max, default)}}
    used to build the What-if input form dynamically from the real data,
    so we never hardcode category lists that could drift from the data.
    """
    schema = {}
    for col in sample_df.columns:
        if col == TARGET_COLUMN or col in DERIVED_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(sample_df[col]):
            schema[col] = {
                "type": "numeric",
                "min": float(sample_df[col].min()),
                "max": float(sample_df[col].max()),
                "default": float(sample_df[col].median()),
            }
        else:
            schema[col] = {
                "type": "categorical",
                "options": sorted(sample_df[col].dropna().unique().tolist()),
                "default": sample_df[col].mode()[0],
            }
    return schema
