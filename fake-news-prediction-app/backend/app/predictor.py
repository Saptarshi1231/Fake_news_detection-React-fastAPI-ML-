"""Load the trained classifier and serve predictions to the FastAPI layer."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from .preprocessing import stemming

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "fake_news_model.joblib"


class ModelNotTrainedError(RuntimeError):
    """Raised when the model artifact has not been created yet."""


@lru_cache(maxsize=1)
def load_artifact() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise ModelNotTrainedError(
            "The model is not trained yet. Run `python backend/prepare_dataset.py`, "
            "then `python backend/train_model.py` from the project root."
        )
    try:
        artifact = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the model at {MODEL_PATH}. Retrain it with `python backend/train_model.py`."
        ) from exc
    required = {"model", "vectorizer", "stop_words", "metrics"}
    missing = required.difference(artifact)
    if missing:
        raise RuntimeError(f"Model artifact is missing fields: {', '.join(sorted(missing))}")
    return artifact


def model_status() -> bool:
    try:
        load_artifact()
        return True
    except (ModelNotTrainedError, RuntimeError, OSError):
        return False


def get_model_metrics() -> dict[str, Any]:
    artifact = load_artifact()
    return {
        "model_version": artifact.get("model_version", "unknown"),
        "trained_at_utc": artifact.get("trained_at_utc"),
        "features": artifact.get("features", "title + text"),
        "metrics": artifact["metrics"],
        "label_mapping": artifact.get("label_mapping", {0: "Real News", 1: "Fake News"}),
        "note": "These are historical hold-out metrics, not proof of current-news fact-checking accuracy.",
    }


def predict_news(title: str, text: str = "") -> dict[str, Any]:
    artifact = load_artifact()
    document = f"{title} {text}".strip()
    processed = stemming(document, set(artifact["stop_words"]))
    features = artifact["vectorizer"].transform([processed])
    model = artifact["model"]
    prediction = int(model.predict(features)[0])

    probabilities = model.predict_proba(features)[0]
    class_probabilities = {
        int(label): float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }
    probability_real = class_probabilities.get(0, 0.0)
    probability_fake = class_probabilities.get(1, 0.0)

    return {
        "prediction": prediction,
        "label": "Fake News" if prediction == 1 else "Real News",
        "probability_fake": probability_fake,
        "probability_real": probability_real,
        "confidence": max(probability_fake, probability_real),
        "model_note": "A model estimate based on historical training data; not an independent fact-check.",
    }
