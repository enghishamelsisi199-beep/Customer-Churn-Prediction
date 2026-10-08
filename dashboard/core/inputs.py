import pandas as pd
import streamlit as st


def render_input_form(schema: dict) -> pd.DataFrame:
    """Renders a form built from the feature schema and returns a
    single-row DataFrame with whatever the user picked/entered."""
    values = {}

    cols = st.columns(3)
    for i, (feature, spec) in enumerate(schema.items()):
        with cols[i % 3]:
            if spec["type"] == "numeric":
                values[feature] = st.number_input(
                    feature,
                    min_value=spec["min"],
                    max_value=spec["max"],
                    value=spec["default"],
                )
            else:
                options = spec["options"]
                default_index = options.index(spec["default"]) if spec["default"] in options else 0
                values[feature] = st.selectbox(feature, options=options, index=default_index)

    return pd.DataFrame([values])
