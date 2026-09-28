from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from src.audio.augment import add_noise_at_snr
from src.audio.features import extract_features_from_signal, log_mel_from_signal
from src.audio.preprocess import load_audio
from src.ml.dataset import load_feature_cache, read_split
from src.ml.evaluation import metrics_from_predictions, save_confusion_matrix, save_metrics
from src.ml.training_config import (
    CLASSES,
    INDEX_TO_CLASS,
    MODELS_DIR,
    REPORTS_DIR,
    ROOT,
    SEED,
)


def evaluate_classical(model_key: str, display_name: str):
    model_path = MODELS_DIR / f"{model_key}.joblib"
    if not model_path.exists():
        return None
    bundle = joblib.load(model_path)
    model = bundle["model"]
    X_test, y_test, _ = load_feature_cache("test")
    pred = model.predict(X_test)
    metrics = metrics_from_predictions(y_test, pred, display_name)
    save_metrics(metrics, REPORTS_DIR, f"{model_key}_test")
    save_confusion_matrix(
        y_test,
        pred,
        REPORTS_DIR / f"{model_key}_test_confusion_matrix.png",
        f"{display_name} - Test Confusion Matrix",
    )
    return metrics


def evaluate_cnn(batch_size: int = 16):
    model_path = MODELS_DIR / "custom_cnn.keras"
    if not model_path.exists():
        return None
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not installed; skipping CNN test evaluation.")
        return None

    from src.ml.cnn_data import AudioSequence

    test_df = read_split("test")
    sequence = AudioSequence(test_df, batch_size=batch_size, shuffle=False, augment=False, seed=SEED)
    model = tf.keras.models.load_model(model_path)
    probs = model.predict(sequence, verbose=1)
    pred_indices = np.argmax(probs, axis=1)
    y_pred = np.asarray([INDEX_TO_CLASS[int(i)] for i in pred_indices])
    y_true = test_df["class_label"].astype(str).to_numpy()
    metrics = metrics_from_predictions(y_true, y_pred, "Custom CNN")
    save_metrics(metrics, REPORTS_DIR, "custom_cnn_test")
    save_confusion_matrix(
        y_true,
        y_pred,
        REPORTS_DIR / "custom_cnn_test_confusion_matrix.png",
        "Custom CNN - Test Confusion Matrix",
    )
    return metrics


def noisy_predictions_classical(model_key: str, snr_db: float, test_df: pd.DataFrame):
    bundle = joblib.load(MODELS_DIR / f"{model_key}.joblib")
    model = bundle["model"]
    rng = np.random.default_rng(SEED + int(snr_db * 10))
    X, y_true = [], []
    for _, row in test_df.iterrows():
        signal, sr = load_audio(str(ROOT / row["relative_path"]))
        noisy = add_noise_at_snr(signal, rng, snr_db)
        X.append(extract_features_from_signal(noisy, sr))
        y_true.append(str(row["class_label"]))
    pred = model.predict(np.vstack(X))
    return np.asarray(y_true), np.asarray(pred)


def noisy_predictions_cnn(snr_db: float, test_df: pd.DataFrame, batch_size: int = 16):
    try:
        import tensorflow as tf
    except ImportError:
        return None
    model = tf.keras.models.load_model(MODELS_DIR / "custom_cnn.keras")
    rng = np.random.default_rng(SEED + int(snr_db * 10))
    y_true = []
    specs = []
    predictions = []

    for _, row in test_df.iterrows():
        signal, sr = load_audio(str(ROOT / row["relative_path"]))
        noisy = add_noise_at_snr(signal, rng, snr_db)
        specs.append(log_mel_from_signal(noisy, sr)[..., np.newaxis])
        y_true.append(str(row["class_label"]))
        if len(specs) >= batch_size:
            probs = model.predict(np.asarray(specs, dtype=np.float32), verbose=0)
            predictions.extend(np.argmax(probs, axis=1).tolist())
            specs.clear()
    if specs:
        probs = model.predict(np.asarray(specs, dtype=np.float32), verbose=0)
        predictions.extend(np.argmax(probs, axis=1).tolist())

    y_pred = np.asarray([INDEX_TO_CLASS[int(i)] for i in predictions])
    return np.asarray(y_true), y_pred


def run_noise_robustness(batch_size: int = 16):
    test_df = read_split("test")
    snr_levels = [20.0, 10.0, 5.0]
    rows = []

    for model_key, display_name in (("random_forest", "Random Forest"), ("svm", "SVM")):
        if not (MODELS_DIR / f"{model_key}.joblib").exists():
            continue
        for snr in snr_levels:
            print(f"Noise test: {display_name}, SNR={snr:g} dB")
            y_true, y_pred = noisy_predictions_classical(model_key, snr, test_df)
            metrics = metrics_from_predictions(y_true, y_pred, display_name)
            rows.append({
                "model": display_name,
                "snr_db": snr,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "critical_event_recall_mean": metrics["critical_event_recall_mean"],
            })

    if (MODELS_DIR / "custom_cnn.keras").exists():
        for snr in snr_levels:
            print(f"Noise test: Custom CNN, SNR={snr:g} dB")
            result = noisy_predictions_cnn(snr, test_df, batch_size=batch_size)
            if result is None:
                break
            y_true, y_pred = result
            metrics = metrics_from_predictions(y_true, y_pred, "Custom CNN")
            rows.append({
                "model": "Custom CNN",
                "snr_db": snr,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "critical_event_recall_mean": metrics["critical_event_recall_mean"],
            })

    if rows:
        noise_df = pd.DataFrame(rows)
        noise_df.to_csv(REPORTS_DIR / "noise_robustness.csv", index=False)
        summary = noise_df.groupby("model", as_index=False).agg(
            noise_accuracy_mean=("accuracy", "mean"),
            noise_macro_f1_mean=("macro_f1", "mean"),
            noise_critical_recall_mean=("critical_event_recall_mean", "mean"),
        )
        summary.to_csv(REPORTS_DIR / "noise_robustness_summary.csv", index=False)
        print(f"Noise robustness report: {REPORTS_DIR / 'noise_robustness_summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained SonicSentinel models on unseen test audio.")
    parser.add_argument("--noise", action="store_true", help="Also test 20/10/5 dB noise robustness.")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for key, display in (("random_forest", "Random Forest"), ("svm", "SVM")):
        metrics = evaluate_classical(key, display)
        if metrics:
            results.append({k: v for k, v in metrics.items() if k != "class_report"})

    cnn_metrics = evaluate_cnn(args.batch_size)
    if cnn_metrics:
        results.append({k: v for k, v in cnn_metrics.items() if k != "class_report"})

    if results:
        pd.DataFrame(results).to_csv(REPORTS_DIR / "test_model_metrics.csv", index=False)
        print("\nTest-set model metrics")
        print(pd.DataFrame(results)[["model", "accuracy", "macro_f1", "critical_event_recall_mean"]].to_string(index=False))
    else:
        raise SystemExit("No trained models found. Train RF/SVM/CNN first.")

    if args.noise:
        run_noise_robustness(args.batch_size)


if __name__ == "__main__":
    main()
