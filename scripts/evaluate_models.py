"""Fresh features, identical inference label rule, validation/test kept separate."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.audio.preprocess import load_audio_native, preprocess_signal
from src.audio.features import extract_features_from_signal, log_mel_from_signal
from src.audio.quality import assess_quality
from src.audio.augment import add_noise_at_snr
from src.ml.dataset import read_split
from src.ml.evaluation import metrics_from_predictions, save_metrics, save_confusion_matrix
from src.ml.predict import score_arrays, model_path, model_digest, MODEL_KEYS
from src.ml.reliability import (
    format_probabilities, decision_for, read_policy, sha256_file,
    SRS_FOLDERS, CRITICAL_RECALL_CLASSES,
)
from src.ml.training_config import CLASSES, MODELS_DIR, REPORTS_DIR, DURATION_SECONDS, SEED

DISPLAY = {"random_forest": "Random Forest", "svm": "SVM", "custom_cnn": "Custom CNN"}


def srs_metric_status(metrics):
    report = metrics["class_report"]
    critical = {label: report.get(label, {}).get("recall") for label in sorted(CRITICAL_RECALL_CLASSES)}
    missing = [label for label in SRS_FOLDERS.values() if label not in CLASSES]
    return {
        "all_ten_classes": not missing,
        "untrained_srs_classes": missing,
        "accuracy_at_least_0_85": metrics["accuracy"] >= 0.85,
        "macro_f1_at_least_0_80": metrics["macro_f1"] >= 0.80,
        "critical_recall_per_class": critical,
        "every_critical_class_recall_at_least_0_85": all(v is not None and v >= .85 for v in critical.values()),
        "note": "Metric checks only; this does not certify complete SRS compliance or GTM performance.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--noise", action="store_true", help="Also evaluate reproducible white noise at 20, 10, 5 dB SNR.")
    parser.add_argument("--batch-size", type=int, default=16, help="Retained for CLI compatibility; CNN predict uses Keras batches.")
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS))
    args = parser.parse_args()
    for key in args.models:
        if not model_path(key).is_file():
            raise SystemExit(f"Missing {model_path(key)}. No model was silently skipped.")

    df = read_split(args.split)
    if set(df["class_label"]) != set(CLASSES):
        raise SystemExit("Split classes differ from training_config.CLASSES. Check that code, splits and models come from the same run.")
    # Refuse obvious leakage before measuring accuracy. Metadata should provide
    # source-level IDs for edited/transcoded files that byte hashes cannot match.
    train = read_split("train")
    for column in ("relative_path", "audio_id", "original_audio_id", "sha256", "source_sha256", "pcm_sha256", "split_group_id"):
        if column in train and column in df:
            a = set(train[column].dropna().astype(str)) - {""}
            b = set(df[column].dropna().astype(str)) - {""}
            if a & b:
                raise SystemExit(f"Train/{args.split} overlap in {column}; fix the split before evaluation.")

    digest = hashlib.sha256()
    signals, qualities, durations = [], [], []
    print(f"Reading {len(df)} {args.split} originals; rebuilding features from audio.", flush=True)
    for number, (_, row) in enumerate(df.iterrows(), 1):
        relative = str(row["relative_path"]).replace("\\", "/")
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise SystemExit(f"Missing or invalid audio path: {relative}")
        actual_hash = sha256_file(path)
        if "sha256" in row and pd.notna(row["sha256"]) and str(row["sha256"]) and str(row["sha256"]) != actual_hash:
            raise SystemExit(f"Audio changed after split creation: {relative}. Rebuild the dataset in a new training run.")
        digest.update(f"{relative}|{row['class_label']}|{actual_hash}\n".encode())
        native, sr = load_audio_native(str(path))
        if not np.all(np.isfinite(native)):
            raise SystemExit(f"Non-finite audio samples: {relative}")
        qualities.append(assess_quality(native)["label"])
        durations.append(len(native) / sr)
        signal, sr = preprocess_signal(native, sr)
        signals.append(signal)
        if number % 25 == 0:
            print(f"Read {number}/{len(df)}", flush=True)
    fingerprint = digest.hexdigest()
    policy = read_policy(ROOT)
    y_true = df["class_label"].astype(str).to_numpy()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows, noise_rows = [], []
    scenarios = [None] + ([20, 10, 5] if args.noise else [])

    for snr in scenarios:
        print("Extracting features:", "clean" if snr is None else f"SNR {snr} dB", flush=True)
        if snr is None:
            current = signals
        else:
            rng = np.random.default_rng(SEED + snr * 10)
            current = [add_noise_at_snr(signal, rng, snr) for signal in signals]
        X = np.vstack([extract_features_from_signal(x, sr) for x in current]) if any(k != "custom_cnn" for k in args.models) else None
        mels = np.asarray([log_mel_from_signal(x, sr)[..., None] for x in current]) if "custom_cnn" in args.models else None
        for key in args.models:
            labels, probabilities = score_arrays(key, X, mels)
            if set(labels) != set(CLASSES):
                raise SystemExit(f"{key} has a different class set from the split. Copy the matching model export.")
            formatted = [format_probabilities(labels, p) for p in probabilities]
            y_pred = [item["prediction"]["label"] for item in formatted]
            metrics = metrics_from_predictions(y_true, y_pred, DISPLAY[key])
            metrics.update({"split": args.split, "decision_rule": "argmax_predict_proba",
                            "model_sha256": model_digest(key), "split_fingerprint": fingerprint,
                            "class_labels": labels})
            if key == "custom_cnn":
                metrics["cnn_manifest_sha256"] = sha256_file(MODELS_DIR / "custom_cnn_manifest.json")
            if snr is not None:
                noise_rows.append({"model": DISPLAY[key], "snr_db": snr, "split": args.split,
                                   "model_sha256": metrics["model_sha256"], "split_fingerprint": fingerprint,
                                   **{m: metrics[m] for m in ("accuracy", "macro_f1", "critical_event_recall_mean")}})
                continue
            decisions = [decision_for(item, quality, policy,
                         ["clip_exceeds_model_window"] if dur > DURATION_SECONDS + .01 else [])
                         for item, quality, dur in zip(formatted, qualities, durations)]
            accepted = np.asarray([not d["manual_review_required"] for d in decisions])
            correct = np.asarray(y_pred) == y_true
            metrics["policy_assessment"] = {
                "policy": policy,
                "accepted_fraction": float(accepted.mean()),
                "manual_review_fraction": float(1 - accepted.mean()),
                "accepted_accuracy": float(correct[accepted].mean()) if accepted.any() else None,
                "accepted_count": int(accepted.sum()),
                "total_count": len(df),
                "critical_recall_after_review_gate": {
                    label: (float((correct & accepted & (y_true == label)).sum() / (y_true == label).sum())
                            if (y_true == label).any() else None)
                    for label in sorted(CRITICAL_RECALL_CLASSES)
                },
                "note": "Rejected samples remain in raw accuracy/F1 denominators; gate recall counts rejected critical samples as misses.",
            }
            metrics["srs_metric_status"] = srs_metric_status(metrics)
            save_metrics(metrics, REPORTS_DIR, f"{key}_{args.split}")
            save_confusion_matrix(y_true, y_pred, REPORTS_DIR / f"{key}_{args.split}_confusion_matrix.png", f"{DISPLAY[key]} - {args.split}")
            predictions = []
            for i, item in enumerate(formatted):
                predictions.append({"audio_id": df.iloc[i]["audio_id"], "actual_class": y_true[i],
                    "predicted_class": y_pred[i], "correct": bool(correct[i]),
                    "confidence": item["prediction"]["confidence"], "top_two_margin": item["top_two_margin"],
                    "manual_review": decisions[i]["manual_review_required"],
                    "review_reasons": "|".join(decisions[i]["reasons"]),
                    **{f"score::{label}": item["all_confidences"][label] for label in labels}})
            pd.DataFrame(predictions).to_csv(REPORTS_DIR / f"{key}_{args.split}_predictions.csv", index=False)
            rows.append({"model": DISPLAY[key], **{m: metrics[m] for m in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "critical_event_recall_mean")}})
            print(f"{DISPLAY[key]}: accuracy={metrics['accuracy']:.4f}, macro F1={metrics['macro_f1']:.4f}", flush=True)

    pd.DataFrame(rows).to_csv(REPORTS_DIR / f"{args.split}_model_metrics.csv", index=False)
    if noise_rows:
        noise = pd.DataFrame(noise_rows)
        noise.to_csv(REPORTS_DIR / f"{args.split}_noise_robustness.csv", index=False)
        summary = noise.groupby(["model", "model_sha256", "split_fingerprint"], as_index=False).agg(
            noise_accuracy_mean=("accuracy", "mean"), noise_macro_f1_mean=("macro_f1", "mean"),
            noise_critical_recall_mean=("critical_event_recall_mean", "mean"))
        summary.to_csv(REPORTS_DIR / f"{args.split}_noise_robustness_summary.csv", index=False)
    print(f"Reports saved: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
