"""One analysis pipeline for uploaded files, batch uploads and live windows.

decode -> validate -> quality -> Python model (selected 3 s window)
       -> GTM model on the SAME window (independently) -> comparison -> final decision.
Everything returned is plain JSON (no NumPy types, no NaN) so it can always be
stored in MySQL/SQLite JSON columns.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from src.ml.training_config import ROOT

MIN_SECONDS = 0.30
MAX_SECONDS = 60.0


class AudioRejected(ValueError):
    """Raised for invalid/silent/too short recordings (shown to the user)."""


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, Path):
        return str(value)
    return value


def _python_block(result: dict) -> dict:
    return {
        "status": "ready",
        "model": str(result.get("model", "selected")),
        "model_sha256": result.get("model_sha256"),
        "prediction": result["prediction"],
        "top3": result.get("top3", []),
        "all_confidences": result.get("all_confidences", {}),
        "top_two_margin": result.get("top_two_margin"),
        "input_policy": result.get("input_policy"),
        "window_note": result.get("window_note"),
        "rule_decision": result.get("decision", {}),
    }


def analyze_file(path, *, live: bool = False) -> dict:
    from src.audio.preprocess import load_audio_native
    from src.audio.quality import assess_quality
    from src.ml.reliability import read_policy

    started = time.perf_counter()
    path = Path(path)
    try:
        native, sr = load_audio_native(str(path))
    except Exception as exc:  # decoding problems -> clear message
        raise AudioRejected("Could not decode this recording. Check that it is a valid WAV, MP3, FLAC, OGG or M4A file.") from exc
    if not np.all(np.isfinite(native)):
        raise AudioRejected("The recording contains invalid sample values (damaged file).")
    duration = float(len(native) / sr)
    quality = assess_quality(native)
    if not live:
        if duration < MIN_SECONDS:
            raise AudioRejected(f"Recording is too short ({duration:.2f} s). Use at least {MIN_SECONDS:.2f} seconds.")
        if duration > MAX_SECONDS:
            raise AudioRejected(f"Recording is {duration:.0f} s long. Use clips up to {MAX_SECONDS:.0f} seconds (split longer files).")
        if quality["label"] == "Unusable":
            raise AudioRejected("Audio is silent or unusable. Choose a recording with a clear audible signal.")

    policy = read_policy(ROOT)
    python, gtm = None, None
    python_error = gtm_error = None
    segment, seg_sr = None, None

    if quality["label"] != "Unusable":
        try:
            from src.ml.predict import predict_models
            results, segment, seg_sr = predict_models(str(path), ["selected"], return_segment=True)
            python = _python_block(results[0])
        except Exception as exc:
            python_error = str(exc)
            try:
                from src.audio.preprocess import preprocess_signal
                segment, seg_sr = preprocess_signal(native, sr)
            except Exception:
                segment, seg_sr = native, sr
        try:
            from src.ml.gtm import classify_signal
            gtm = classify_signal(segment, seg_sr)       # independent: no Python output is passed
        except Exception as exc:
            gtm_error = str(exc)

    from src.ml.comparison import compare
    decision = compare(python, gtm, quality["label"], policy)
    final = decision["final"]
    if final is None and quality["label"] == "Unusable":
        final = {
            "label": None, "display_label": "Silent / unusable", "confidence": None, "confidence_level": "None",
            "top_two_margin": None, "top3": [], "all_confidences": {}, "method": "not classified",
            "severity": "Informational", "status": "Classified", "alert_status": "No alert",
            "active_alert": False, "candidate_alert": False, "manual_review_required": False,
            "reasons": ["silent_window"], "overlapping_sounds": False,
            "decision_note": "Silent window: no class was forced.",
        }

    payload = {
        "duration_seconds": round(duration, 3),
        "sample_rate": int(sr),
        "quality": quality,
        "python": python or {"status": "unavailable", "error": python_error or "Silent audio was not classified."},
        "gtm": gtm or {"status": "unavailable", "error": gtm_error or "Silent audio was not classified."},
        "comparison": decision["comparison"],
        "final": final,
        "policy_version": policy.get("version"),
        "processing_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    return to_jsonable(payload)
