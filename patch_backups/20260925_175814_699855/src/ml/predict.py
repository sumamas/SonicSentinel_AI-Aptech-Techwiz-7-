from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from src.audio.features import extract_features, log_mel_spectrogram
from src.ml.training_config import CLASSES, INDEX_TO_CLASS, MODELS_DIR


@lru_cache(maxsize=4)
def _load_joblib_model(model_name: str):
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found: {path}")
    return joblib.load(path)


@lru_cache(maxsize=1)
def _load_cnn():
    path = MODELS_DIR / "custom_cnn.keras"
    if not path.exists():
        raise FileNotFoundError(f"Trained CNN not found: {path}")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to run the custom CNN.") from exc
    return tf.keras.models.load_model(path)


def get_selected_model_name() -> str:
    manifest = MODELS_DIR / "selected_model.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return str(payload["selected_model"])
    if (MODELS_DIR / "custom_cnn.keras").exists():
        return "custom_cnn"
    if (MODELS_DIR / "svm.joblib").exists():
        return "svm"
    if (MODELS_DIR / "random_forest.joblib").exists():
        return "random_forest"
    raise FileNotFoundError("No trained SonicSentinel model is available yet.")


def _format_probabilities(labels: list[str], probabilities: np.ndarray) -> dict:
    order = np.argsort(probabilities)[::-1]
    top3 = [
        {"label": labels[int(index)], "confidence": float(probabilities[int(index)])}
        for index in order[:3]
    ]
    return {
        "prediction": top3[0],
        "top3": top3,
        "all_confidences": {
            labels[index]: float(probabilities[index]) for index in range(len(labels))
        },
    }


def predict_file(file_path: str, model_name: str = "selected") -> dict:
    if model_name == "selected":
        model_name = get_selected_model_name()

    if model_name in {"random_forest", "svm"}:
        bundle = _load_joblib_model(model_name)
        model = bundle["model"]
        x = extract_features(file_path).reshape(1, -1)
        probabilities = model.predict_proba(x)[0]
        labels = [str(label) for label in model.classes_]
        result = _format_probabilities(labels, probabilities)
    elif model_name == "custom_cnn":
        model = _load_cnn()
        mel = log_mel_spectrogram(file_path)[np.newaxis, ..., np.newaxis]
        probabilities = model.predict(mel, verbose=0)[0]
        result = _format_probabilities(CLASSES, probabilities)
    else:
        raise ValueError("model_name must be selected, random_forest, svm, or custom_cnn")

    result["model"] = model_name
    return result
