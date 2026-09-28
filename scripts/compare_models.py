"""Select from fresh validation reports. Never read test metrics to select a model."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.ml.training_config import MODELS_DIR, REPORTS_DIR, CLASSES
from src.ml.predict import MODEL_KEYS, model_digest
from src.ml.reliability import sha256_file


def main():
    rows = []
    for key in MODEL_KEYS:
        path = REPORTS_DIR / f"{key}_validation_metrics.json"
        if not path.is_file():
            raise SystemExit("Run python scripts/evaluate_models.py --split validation --noise first.")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if metrics.get("split") != "validation" or metrics.get("decision_rule") != "argmax_predict_proba":
            raise SystemExit(f"Refresh {key} validation evaluation with the patched evaluator.")
        if metrics.get("model_sha256") != model_digest(key) or set(metrics.get("class_labels", [])) != set(CLASSES):
            raise SystemExit(f"Stale validation report for {key}; evaluate the current models again.")
        if key == "custom_cnn" and metrics.get("cnn_manifest_sha256") != sha256_file(MODELS_DIR / "custom_cnn_manifest.json"):
            raise SystemExit("CNN manifest changed; repeat validation evaluation.")
        rows.append({"key": key, **{k: metrics[k] for k in (
            "model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "critical_event_recall_mean",
            "model_sha256", "split_fingerprint")}})
    df = pd.DataFrame(rows)
    if df["split_fingerprint"].nunique() != 1:
        raise SystemExit("Models were evaluated on different validation audio. Evaluate all three together.")
    df["selection_score"] = (.15 * df.accuracy + .10 * df.macro_precision + .15 * df.macro_recall
                             + .35 * df.macro_f1 + .25 * df.critical_event_recall_mean)
    noise_path = REPORTS_DIR / "validation_noise_robustness_summary.csv"
    noise_used = False
    if noise_path.is_file():
        noise = pd.read_csv(noise_path)
        merged = df.merge(noise, on=["model", "model_sha256", "split_fingerprint"], how="left", validate="one_to_one")
        if merged["noise_macro_f1_mean"].notna().all():
            df = merged
            df["selection_score"] = (.85 * df.selection_score + .15 *
                (.5 * df.noise_macro_f1_mean + .5 * df.noise_critical_recall_mean))
            noise_used = True
        else:
            print("Stale/incomplete validation noise report ignored for ALL models.")
    df = df.sort_values(["selection_score", "macro_f1", "critical_event_recall_mean"], ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    selected = df.iloc[0]
    manifest = {
        "selected_model": str(selected["key"]), "display_name": str(selected["model"]),
        "selection_split": "validation", "decision_rule": "argmax_predict_proba",
        "model_sha256": str(selected["model_sha256"]), "split_fingerprint": str(selected["split_fingerprint"]),
        "selection_score": float(selected["selection_score"]), "validation_noise_used": noise_used,
        "selection_method": "Fixed multi-metric validation score; class-wise errors still require review.",
        "classes": CLASSES,
    }
    if selected["key"] == "custom_cnn":
        manifest["cnn_manifest_sha256"] = sha256_file(MODELS_DIR / "custom_cnn_manifest.json")
    (MODELS_DIR / "selected_model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(df[["rank", "model", "accuracy", "macro_f1", "critical_event_recall_mean", "selection_score"]].to_string(index=False))
    print("Selected:", manifest["display_name"], "using validation only.")


if __name__ == "__main__":
    main()
