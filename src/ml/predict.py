from __future__ import annotations

import json
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from src.ml.training_config import MODELS_DIR, ROOT, SAMPLE_RATE, DURATION_SECONDS, N_MELS, TARGET_MEL_FRAMES
from src.ml.provenance import assert_input_contract
from src.ml.training_config import CLASSES
from src.ml.reliability import (
    SRS_FOLDERS, sha256_file, read_policy, format_probabilities, decision_for,
)

MODEL_KEYS = ("random_forest", "svm", "custom_cnn")


def model_path(name):
    if name not in MODEL_KEYS:
        raise ValueError(f"Unsupported model: {name}")
    return MODELS_DIR / ("custom_cnn.keras" if name == "custom_cnn" else f"{name}.joblib")


def _file_token(path):
    if not path.is_file():
        raise FileNotFoundError(f"Trained model not found: {path}")
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=6)
def _load_bundle_cached(path, modified, size):
    from sklearn.exceptions import InconsistentVersionWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        try:
            bundle = joblib.load(path)
        except InconsistentVersionWarning as exc:
            raise RuntimeError("scikit-learn version differs from the saved model. Match its training environment.") from exc
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Expected a SonicSentinel joblib bundle with a 'model' key.")
    labels = [str(x) for x in bundle["model"].classes_]
    if "classes" in bundle and set(bundle["classes"]) != set(labels):
        raise ValueError("Bundle classes disagree with the fitted estimator. Copy a consistent model export.")
    assert_input_contract(bundle.get("input_contract"))
    if set(labels)!=set(CLASSES):raise ValueError("Expected this run's ten active classes.")
    return bundle


def _load_joblib_model(name):
    return _load_bundle_cached(*_file_token(model_path(name)))


def cnn_labels():
    path = MODELS_DIR / "custom_cnn_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Missing custom_cnn_manifest.json; copy it together with custom_cnn.keras.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert_input_contract(manifest.get("input_contract"), cnn=True)
    labels = manifest.get("classes")
    if not isinstance(labels, list) or len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("CNN manifest has no valid ordered classes list.")
    if manifest.get("class_to_index") != {label: i for i, label in enumerate(labels)}:
        raise ValueError("CNN manifest class order and class_to_index disagree.")
    if manifest.get("sample_rate") != SAMPLE_RATE or manifest.get("duration_seconds") != DURATION_SECONDS:
        raise ValueError("CNN training preprocessing differs from the current sample rate/duration.")
    if manifest.get("model_sha256") and manifest["model_sha256"] != model_digest("custom_cnn"):
        raise ValueError("CNN model and manifest were exported from different runs.")
    return labels


@lru_cache(maxsize=2)
def _load_cnn_cached(path, modified, size):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to run the custom CNN.") from exc
    return tf.keras.models.load_model(path, compile=False)


def _load_cnn():
    labels = cnn_labels()
    model = _load_cnn_cached(*_file_token(model_path("custom_cnn")))
    if int(model.output_shape[-1]) != len(labels):
        raise ValueError("CNN output count and saved manifest disagree.")
    if tuple(model.input_shape[1:]) != (N_MELS, TARGET_MEL_FRAMES, 1):
        raise ValueError("CNN input shape differs from current feature preprocessing.")
    return model


@lru_cache(maxsize=6)
def _digest_cached(path, modified, size):
    return sha256_file(Path(path))


def model_digest(name):
    return _digest_cached(*_file_token(model_path(name)))


def get_selected_model_name():
    path = MODELS_DIR / "selected_model.json"
    instruction = "Run python scripts/evaluate_models.py --split validation, then python scripts/compare_models.py."
    if not path.is_file():
        raise RuntimeError("No validated model selection. " + instruction)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("selection_split") != "validation":
        raise RuntimeError("Old/test-based model selection needs refreshing. " + instruction)
    key = str(payload["selected_model"])
    if payload.get("model_sha256") != model_digest(key):
        raise RuntimeError("Selected model has changed since evaluation. " + instruction)
    if key == "custom_cnn" and payload.get("cnn_manifest_sha256") != sha256_file(MODELS_DIR / "custom_cnn_manifest.json"):
        raise RuntimeError("CNN label manifest has changed since evaluation. " + instruction)
    return key


def score_arrays(name, features=None, mels=None):
    if name in {"svm", "random_forest"}:
        bundle = _load_joblib_model(name)
        model = bundle["model"]
        if features is None:
            raise ValueError("Classical features were not supplied.")
        expected = int(getattr(model, "n_features_in_", bundle.get("feature_count", -1)))
        if features.shape[1] != expected:
            raise ValueError(f"Feature mismatch: model expects {expected}, received {features.shape[1]}.")
        return [str(x) for x in model.classes_], np.asarray(model.predict_proba(features))
    if name == "custom_cnn":
        return cnn_labels(), np.asarray(_load_cnn().predict(mels, verbose=0))
    raise ValueError(f"Unsupported model: {name}")


def predict_models(file_path, names=MODEL_KEYS, return_segment=False):
    from src.audio.preprocess import load_audio_native, preprocess_signal
    from src.audio.features import extract_features_from_signal, log_mel_from_signal
    from src.audio.quality import assess_quality

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    names = list(dict.fromkeys(get_selected_model_name() if n == "selected" else n for n in names))
    if not names or any(n not in MODEL_KEYS for n in names):
        raise ValueError("Choose selected, random_forest, svm or custom_cnn.")
    native, sr = load_audio_native(str(path))
    if not np.all(np.isfinite(native)):
        raise ValueError("Audio contains invalid sample values.")
    duration = float(len(native) / sr)
    if duration < 0.30:
        raise ValueError("Audio is shorter than 0.30 seconds.")
    if duration > 60:
        raise ValueError("Audio exceeds 60 seconds. Split and inspect the recording first.")
    quality = assess_quality(native)  # before trimming, peak normalization or padding
    if quality["label"] == "Unusable":
        raise ValueError("Audio is silent or unusable; a class was not forced.")
    signal, model_sr = preprocess_signal(native, sr)
    X = (extract_features_from_signal(signal, model_sr).reshape(1, -1)
         if any(n != "custom_cnn" for n in names) else None)
    mel = (log_mel_from_signal(signal, model_sr)[None, ..., None]
           if "custom_cnn" in names else None)
    policy = read_policy(ROOT)
    # V5 uses one highest-energy 3-second window. A longer clip is NOT a reason
    # for manual review by itself (that bug sent every >3 s file to "Uncertain");
    # it is reported as an informational note instead.
    extra = []
    window_note = ("Longer clip: the loudest 3-second window was classified." if duration > DURATION_SECONDS + 0.01 else None)
    results = []
    for name in names:
        labels, probabilities = score_arrays(name, X, mel)
        result = format_probabilities(labels, probabilities[0])
        result.update({
            "model": name,
            "model_sha256": model_digest(name),
            "supported_classes": labels,
            "untrained_srs_classes": [x for x in SRS_FOLDERS.values() if x not in labels],
            "quality": quality,
            "audio_duration_seconds": duration,
            "input_policy": "v5: highest-energy trimmed 3-second window; same training and inference preprocessing",
        })
        result["decision"] = decision_for(result, quality["label"], policy, extra)
        result["window_note"] = window_note
        results.append(result)
    if return_segment:
        return results, signal, model_sr
    return results


def predict_file(file_path, model_name="selected"):
    return predict_models(file_path, [model_name])[0]


def compare_predictions(results):
    """Comparison only: no voting, relabeling or invented ensemble probabilities."""
    labels = {x["prediction"]["label"] for x in results}
    disagreement = len(labels) > 1
    review = disagreement or any(x["decision"]["manual_review_required"] for x in results)
    return {
        "models_disagree": disagreement,
        "manual_review_required": review,
        "status": "Manual Review" if review else "Models agree",
        "gtm_comparison": "not performed; these are three Python models",
        "results": results,
    }
