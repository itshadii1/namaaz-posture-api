import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent.parent / "ml" / "body_language.pkl"
CONFIDENCE_THRESHOLD = 0.5

# Column names matching the training CSV (x1,y1,z1,v1, ..., x33,y33,z33,v33)
FEATURE_COLUMNS: list[str] = [
    coord
    for i in range(1, 34)
    for coord in (f"x{i}", f"y{i}", f"z{i}", f"v{i}")
]

_LABEL_MAP: dict[str, str] = {
    "sajdah": "sajdah",
    "rukooh": "rukooh",
    "qiam": "qiam",
    "julsa": "julsa",
}

_model = None


def load_model():
    global _model
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)

    # Log expected feature count for debugging
    try:
        scaler = _model.steps[0][1]
        if hasattr(scaler, "n_features_in_"):
            logger.info("Model expects %d features", scaler.n_features_in_)
    except Exception:
        pass

    logger.info("Posture model loaded from %s", MODEL_PATH)


def get_model():
    if _model is None:
        load_model()
    return _model


def predict_posture(features: np.ndarray) -> tuple[str, float]:
    model = get_model()
    X = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    label = str(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    confidence = float(np.max(proba))

    normalised = _LABEL_MAP.get(label.lower(), label.lower())

    if confidence < CONFIDENCE_THRESHOLD:
        return "unknown", confidence

    return normalised, confidence
