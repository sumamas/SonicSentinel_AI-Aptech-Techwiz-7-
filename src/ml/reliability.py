"""Auditable rules. Scores are model outputs, not verified correctness rates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SRS_FOLDERS = {
    "machinery_fault": "Machinery Fault",
    "glass_breaking": "Glass Breaking",
    "alarm_siren": "Alarm or Siren",
    "vehicle_horn": "Vehicle Horn",
    "animal_sound": "Animal Sound",
    "gunshot": "Gunshot",
    "panic_scream": "Panic Scream",
    "aggression": "Aggression",
    "person_asking_for_help": "Person Asking for Help",
    "background_noise": "Background Noise",
}
CRITICAL_RECALL_CLASSES = {
    "Gunshot", "Glass Breaking", "Panic Scream", "Aggression", "Person Asking for Help",
}
SEVERITY = {
    "Machinery Fault": "High", "Glass Breaking": "High", "Alarm or Siren": "High",
    "Vehicle Horn": "Low", "Animal Sound": "Low", "Gunshot": "Critical",
    "Panic Scream": "Critical", "Aggression": "High",
    "Person Asking for Help": "Critical", "Background Noise": "Informational",
}
DEFAULT_POLICY = {
    "version": "reliability-patch-1",
    "validation_calibrated": False,
    "min_confidence": 0.60,
    "min_margin": 0.15,
    "alert_min_confidence": 0.80,
    "alert_min_margin": 0.20,
    "note": "Provisional development rules; measure coverage and errors on validation before deployment.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_policy(root: Path) -> dict:
    policy = dict(DEFAULT_POLICY)
    path = Path(root) / "config" / "model_policy.json"
    if path.exists():
        supplied = json.loads(path.read_text(encoding="utf-8"))
        policy.update(supplied)
    for key in ("min_confidence", "min_margin", "alert_min_confidence", "alert_min_margin"):
        value = float(policy[key])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid model policy {key}: expected a number in [0, 1].")
        policy[key] = value
    return policy


def validate_probabilities(labels, probabilities):
    labels = [str(label) for label in labels]
    p = np.asarray(probabilities, dtype=float)
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("Model labels must contain at least two unique classes.")
    if p.ndim != 1 or len(p) != len(labels):
        raise ValueError("Model output size does not match its saved label list.")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Model returned invalid confidence values.")
    if not np.isclose(p.sum(), 1.0, atol=1e-3):
        raise ValueError("Model confidences do not sum to one.")
    return labels, p


def format_probabilities(labels, probabilities) -> dict:
    labels, p = validate_probabilities(labels, probabilities)
    order = np.argsort(-p, kind="stable")
    top = [{"label": labels[i], "confidence": float(p[i])} for i in order[:3]]
    return {
        "prediction": top[0],
        "top3": top,
        "all_confidences": dict(zip(labels, map(float, p))),
        "top_two_margin": float(p[order[0]] - p[order[1]]),
    }


def decision_for(result: dict, quality_label="Good", policy=None, extra_reasons=()) -> dict:
    policy = dict(DEFAULT_POLICY if policy is None else policy)
    predicted = result["prediction"]
    label = predicted["label"]
    confidence = float(predicted["confidence"])
    margin = float(result["top_two_margin"])
    reasons = list(extra_reasons)
    if quality_label not in {"Good", "Acceptable"}:
        reasons.append("audio_quality_requires_review")
    if confidence < policy["min_confidence"]:
        reasons.append("low_confidence")
    if margin < policy["min_margin"]:
        reasons.append("close_top_two_scores")
    candidate_severity = SEVERITY.get(label, "Informational")
    critical_candidate = candidate_severity in {"High", "Critical"}
    if critical_candidate:
        if confidence < policy["alert_min_confidence"]:
            reasons.append("below_alert_confidence")
        if margin < policy["alert_min_margin"]:
            reasons.append("below_alert_margin")
    reasons = sorted(set(reasons))
    review = bool(reasons)
    active = critical_candidate and not review
    return {
        "label": "Uncertain" if review else label,
        "status": "Manual Review" if review else ("Alert Generated" if active else "Classified"),
        "manual_review_required": review,
        "reasons": reasons,
        "severity": "Medium" if review else candidate_severity,
        "candidate_severity": candidate_severity,
        "active_alert": active,
        "policy_version": policy["version"],
        "validation_calibrated": bool(policy.get("validation_calibrated", False)),
        "decision_note": (
            "Uncertain result; review required: " + ", ".join(reasons) + "."
            if review else "Python model passed the configured development rules."
        ),
    }


def probability_labels(model, X):
    """Use the SAME label rule as inference (including for SVC)."""
    p = np.asarray(model.predict_proba(X))
    labels = np.asarray(model.classes_, dtype=str)
    for row in p:
        validate_probabilities(labels, row)
    return labels[np.argmax(p, axis=1)]


def probability_macro_f1(estimator, X, y):
    from sklearn.metrics import f1_score
    return f1_score(y, probability_labels(estimator, X), average="macro", zero_division=0)
