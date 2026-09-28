from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ml.training_config import MODELS_DIR, REPORTS_DIR


def main():
    metrics_path = REPORTS_DIR / "test_model_metrics.csv"
    if not metrics_path.exists():
        raise SystemExit("Run scripts/evaluate_models.py first.")

    df = pd.read_csv(metrics_path)
    required = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "critical_event_recall_mean"]
    for column in required:
        if column not in df.columns:
            raise SystemExit(f"Missing metric column: {column}")

    noise_path = REPORTS_DIR / "noise_robustness_summary.csv"
    if noise_path.exists():
        noise = pd.read_csv(noise_path)
        df = df.merge(noise, on="model", how="left")
    else:
        df["noise_accuracy_mean"] = pd.NA
        df["noise_macro_f1_mean"] = pd.NA
        df["noise_critical_recall_mean"] = pd.NA

    # Multi-metric selection: clean-set metrics carry 85% of the score.
    # When noise testing exists, robustness contributes the remaining 15%.
    clean_score = (
        0.15 * df["accuracy"]
        + 0.10 * df["macro_precision"]
        + 0.15 * df["macro_recall"]
        + 0.35 * df["macro_f1"]
        + 0.25 * df["critical_event_recall_mean"]
    )

    noise_available = df["noise_macro_f1_mean"].notna()
    df["selection_score"] = clean_score
    if noise_available.any():
        robustness = (
            0.50 * df["noise_macro_f1_mean"].fillna(0)
            + 0.50 * df["noise_critical_recall_mean"].fillna(0)
        )
        df.loc[noise_available, "selection_score"] = (
            0.85 * clean_score[noise_available] + 0.15 * robustness[noise_available]
        )

    df = df.sort_values(
        ["selection_score", "macro_f1", "critical_event_recall_mean"],
        ascending=False,
    ).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    comparison_path = REPORTS_DIR / "model_comparison.csv"
    df.to_csv(comparison_path, index=False)

    selected_display = str(df.iloc[0]["model"])
    key_map = {
        "Random Forest": "random_forest",
        "SVM": "svm",
        "Custom CNN": "custom_cnn",
    }
    selected_key = key_map[selected_display]
    selected = {
        "selected_model": selected_key,
        "display_name": selected_display,
        "selection_score": float(df.iloc[0]["selection_score"]),
        "selection_method": "Multi-metric score using clean test accuracy, macro precision/recall/F1, critical-event recall, plus noise robustness when available.",
        "note": "Final deployment choice should also be reviewed against class-wise confusion matrix and project constraints.",
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "selected_model.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    print("\nModel comparison")
    columns = ["rank", "model", "accuracy", "macro_f1", "critical_event_recall_mean", "selection_score"]
    print(df[columns].to_string(index=False))
    print(f"\nSelected model: {selected_display}")
    print(f"Comparison: {comparison_path}")
    print(f"Manifest: {MODELS_DIR / 'selected_model.json'}")


if __name__ == "__main__":
    main()
