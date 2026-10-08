import joblib

from app.config import settings

print(f"Loading model from: {settings.MODEL_PATH}")
_state = {"pipeline": joblib.load(settings.MODEL_PATH)}
print("Model loaded.")


def get_pipeline():
    return _state["pipeline"]


def set_pipeline(new_pipeline):
    """Hot-swaps the in-memory model after a successful /retrain — no
    server restart needed for the new model to start serving predictions."""
    _state["pipeline"] = new_pipeline
