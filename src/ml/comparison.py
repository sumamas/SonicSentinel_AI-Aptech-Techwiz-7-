"""Python vs Google Teachable Machine comparison and the final event decision.

Rules (SRS Step 11, 17, 19 and FR xxxi-xl):
* Each model's raw scores are stored UNCHANGED. Nothing here edits a model output.
* The final result is a transparent, configurable weighted average of the two
  independent probability vectors (weights in config/model_policy.json). This is
  a documented ensemble step, not a hidden rule and not an invented score.
* Consistency status: Acceptable Match / Weak Match / Model Disagreement /
  Uncertain Result.
"""
from __future__ import annotations

from src.ml.reliability import SEVERITY
from src.ml.training_config import CLASSES

CRITICAL_ALERT_SEVERITIES = {"High", "Critical"}

DEFAULT_COMPARISON_POLICY = {
    "fusion_weights": {"python": 0.65, "gtm": 0.35},
    "acceptable_max_confidence_difference": 0.35,
    "weak_match_floor": 0.30,
    "uncertain_floor": 0.35,
    "unknown_floor": 0.25,
    "overlap_threshold": 0.25,
    "background_quiet_ceiling": 0.35,
    "review_on_model_disagreement": True,
    "require_model_agreement_for_alert": False,
    # SRS Step 15: agreement between the two independent models is a
    # confirmation method, so an agreed critical class needs less confidence.
    "agreement_alert_min_confidence": 0.65,
}


def _policy(policy: dict | None) -> dict:
    merged = dict(DEFAULT_COMPARISON_POLICY)
    if policy:
        merged.update({k: v for k, v in policy.items() if k in DEFAULT_COMPARISON_POLICY})
    w = merged["fusion_weights"] or {}
    py, gt = float(w.get("python", 0.65)), float(w.get("gtm", 0.35))
    total = py + gt if py + gt > 0 else 1.0
    merged["fusion_weights"] = {"python": py / total, "gtm": gt / total}
    return merged


def _scores(result: dict | None) -> dict:
    scores = {c: 0.0 for c in CLASSES}
    if result:
        for label, value in (result.get("all_confidences") or {}).items():
            if label in scores:
                scores[label] = float(value)
    return scores


def _ranked(scores: dict):
    return sorted(scores.items(), key=lambda kv: -kv[1])


def compare(python: dict | None, gtm: dict | None, quality_label: str = "Good", policy: dict | None = None) -> dict:
    """Return comparison metrics + final decision. Inputs are the raw model dicts."""
    base_policy = dict(policy or {})
    p = _policy(base_policy.get("comparison"))
    min_conf = float(base_policy.get("min_confidence", 0.60))
    min_margin = float(base_policy.get("min_margin", 0.15))
    alert_conf = float(base_policy.get("alert_min_confidence", 0.80))
    alert_margin = float(base_policy.get("alert_min_margin", 0.20))

    py_ok = bool(python and python.get("prediction"))
    gtm_ok = bool(gtm and gtm.get("prediction") and gtm.get("status") == "ready")
    py_s, gtm_s = _scores(python), _scores(gtm)
    py_rank, gtm_rank = _ranked(py_s), _ranked(gtm_s)

    comparison = {
        "python_available": py_ok,
        "gtm_available": gtm_ok,
        "python_label": py_rank[0][0] if py_ok else None,
        "python_confidence": py_rank[0][1] if py_ok else None,
        "gtm_label": gtm_rank[0][0] if gtm_ok else None,
        "gtm_confidence": gtm_rank[0][1] if gtm_ok else None,
        "class_match": None,
        "confidence_difference": None,
        "python_top_two_margin": (py_rank[0][1] - py_rank[1][1]) if py_ok else None,
        "gtm_top_two_margin": (gtm_rank[0][1] - gtm_rank[1][1]) if gtm_ok else None,
        "gtm_score_for_python_class": gtm_s.get(py_rank[0][0]) if (py_ok and gtm_ok) else None,
        "python_score_for_gtm_class": py_s.get(gtm_rank[0][0]) if (py_ok and gtm_ok) else None,
        "status": "Python Only" if py_ok and not gtm_ok else ("GTM Only" if gtm_ok and not py_ok else "Unavailable"),
    }

    if py_ok and gtm_ok:
        same = py_rank[0][0] == gtm_rank[0][0]
        diff = abs(py_rank[0][1] - gtm_rank[0][1])       # SRS formula
        comparison["class_match"] = same
        comparison["confidence_difference"] = diff
        py_top2 = {py_rank[0][0], py_rank[1][0]}
        gtm_top2 = {gtm_rank[0][0], gtm_rank[1][0]}
        if same and diff <= p["acceptable_max_confidence_difference"] and min(py_rank[0][1], gtm_rank[0][1]) >= p["weak_match_floor"]:
            status = "Acceptable Match"
        elif same or (py_rank[0][0] in gtm_top2 and gtm_rank[0][0] in py_top2):
            status = "Weak Match"
        else:
            status = "Model Disagreement"
        comparison["status"] = status

    # ------------------------------------------------------------ final scores
    if py_ok and gtm_ok:
        w = p["fusion_weights"]
        final_scores = {c: w["python"] * py_s[c] + w["gtm"] * gtm_s[c] for c in CLASSES}
        method = f"weighted average (Python {w['python']:.2f} / GTM {w['gtm']:.2f})"
    elif py_ok:
        final_scores, method = py_s, "Python model only (GTM unavailable)"
    elif gtm_ok:
        final_scores, method = gtm_s, "GTM only (Python model unavailable)"
    else:
        return {"comparison": comparison, "final": None}

    total = sum(final_scores.values()) or 1.0
    final_scores = {k: v / total for k, v in final_scores.items()}
    ranked = _ranked(final_scores)
    label, conf = ranked[0]
    margin = ranked[0][1] - ranked[1][1]

    reasons = []
    if quality_label not in {"Good", "Acceptable"}:
        reasons.append("audio_quality_requires_review")
    overlapping = ranked[1][1] >= p["overlap_threshold"] and ranked[0][1] < 0.70
    quiet_background = label == "Background Noise" and ranked[1][1] < p["background_quiet_ceiling"]
    if not quiet_background:
        if conf < min_conf:
            reasons.append("low_confidence")
        if margin < min_margin:
            reasons.append("close_top_two_scores")
        if overlapping:
            reasons.append("possible_overlapping_sounds")
    if comparison["status"] == "Model Disagreement" and p["review_on_model_disagreement"]:
        reasons.append("model_disagreement")

    severity = SEVERITY.get(label, "Informational")
    critical = severity in CRITICAL_ALERT_SEVERITIES
    agreed = bool(comparison.get("class_match")) and label == comparison.get("python_label")
    if critical:
        needed = min(alert_conf, float(p["agreement_alert_min_confidence"])) if agreed else alert_conf
        if conf < needed:
            reasons.append("below_alert_confidence")
        if margin < alert_margin:
            reasons.append("below_alert_margin")
        if p["require_model_agreement_for_alert"] and comparison["status"] not in {"Acceptable Match", "Weak Match", "Python Only"}:
            reasons.append("critical_without_model_agreement")
    reasons = sorted(set(reasons))
    review = bool(reasons)

    if py_ok and gtm_ok and conf < p["uncertain_floor"]:
        comparison["status"] = "Uncertain Result"
    display = label
    if conf < p["unknown_floor"]:
        display = "Unknown Sound"
        review = True
        reasons = sorted(set(reasons + ["unknown_sound_pattern"]))

    active = critical and not review
    if review:
        status = "Manual Review"
    elif active:
        status = "Alert Generated"
    else:
        status = "Classified"
    confidence_level = "High" if conf >= 0.80 else ("Medium" if conf >= 0.60 else "Low")
    alert_status = "Active alert" if active else ("Pending review" if critical and review else "No alert")

    final = {
        "label": label,
        "display_label": display,
        "confidence": conf,
        "confidence_level": confidence_level,
        "top_two_margin": margin,
        "top3": [{"label": k, "confidence": v} for k, v in ranked[:3]],
        "all_confidences": final_scores,
        "method": method,
        "severity": severity,
        "status": status,
        "alert_status": alert_status,
        "active_alert": active,
        "candidate_alert": critical,
        "manual_review_required": review,
        "reasons": reasons,
        "overlapping_sounds": bool(overlapping),
        "decision_note": ("Review required: " + ", ".join(r.replace('_', ' ') for r in reasons) + "."
                          if review else f"Final decision passed the configured rules ({comparison['status']})."),
    }
    return {"comparison": comparison, "final": final}
