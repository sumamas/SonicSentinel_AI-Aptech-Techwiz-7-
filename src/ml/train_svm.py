from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from src.ml.group_cv import original_group_folds
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone

from src.ml.provenance import input_contract, training_evidence
from src.ml.dataset import load_feature_cache
from src.ml.reliability import probability_labels, probability_macro_f1, sha256_file
from src.ml.evaluation import metrics_from_predictions, save_confusion_matrix, save_metrics
from src.ml.training_config import CLASSES, MODELS_DIR, REPORTS_DIR, SEED


def main():
    parser = argparse.ArgumentParser(description="Train SonicSentinel SVM baseline.")
    parser.add_argument("--quick", action="store_true", help="Use a smaller tuning grid for development runs.")
    args = parser.parse_args()

    X_train, y_train, train_ids = load_feature_cache("train")
    X_val, y_val, _ = load_feature_cache("validation")

    if set(y_train) != set(CLASSES) or set(y_val) != set(CLASSES):
        raise SystemExit("Feature labels differ from active classes; rebuild caches.")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=False, class_weight="balanced", random_state=SEED)),
    ])
    grid = {
        "svc__C": [3, 10] if args.quick else [1, 3, 10, 30],
        "svc__gamma": ["scale", 0.01] if args.quick else ["scale", 0.01, 0.001],
    }
    cv = original_group_folds(y_train, train_ids)
    search = GridSearchCV(pipeline, grid, cv=cv, scoring="f1_macro", n_jobs=int(os.environ.get("SONIC_N_JOBS", "2")), verbose=1)
    search.fit(X_train, y_train)

    # Calibrate on held-out ORIGINALS from train only. SVC internal probability
    # calibration cannot accept our explicit source-group folds.
    model = CalibratedClassifierCV(clone(search.best_estimator_), method="sigmoid", cv=cv, ensemble=True, n_jobs=int(os.environ.get("SONIC_N_JOBS", "2")))
    model.fit(X_train, y_train)
    val_pred = probability_labels(model, X_val)
    metrics = metrics_from_predictions(y_val, val_pred, "SVM")
    metrics["best_params"] = search.best_params_
    metrics["cv_best_macro_f1"] = float(search.best_score_)
    metrics["cv_scoring_rule"] = "Uncalibrated SVC predict; final validation/inference uses calibrated argmax predict_proba."
    metrics["calibration"] = "Three training group-disjoint sigmoid calibrators; original-only held-out folds. Not a guarantee of calibrated field confidence."

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_name": "svm",
        "display_name": "SVM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classes": CLASSES,
        "feature_count": int(X_train.shape[1]),
        "input_contract": input_contract(),
        "best_params": search.best_params_,
        "model": model,
    }
    joblib.dump(bundle, MODELS_DIR / "svm.joblib")

    metrics["training_evidence"] = training_evidence()
    metrics["cv_evaluation"] = "Three group-disjoint folds; held-out folds contain originals only."
    metrics["model_sha256"] = sha256_file(MODELS_DIR / "svm.joblib")
    # Keep CV/tuning evidence when evaluate_models refreshes validation metrics.
    (REPORTS_DIR / "svm_training_summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8",
    )

    save_metrics(metrics, REPORTS_DIR, "svm_validation")
    save_confusion_matrix(
        y_val,
        val_pred,
        REPORTS_DIR / "svm_validation_confusion_matrix.png",
        "SVM - Validation Confusion Matrix",
    )

    print("\nSVM complete")
    print("Best params:", search.best_params_)
    print("Validation accuracy:", round(metrics["accuracy"], 4))
    print("Validation macro F1:", round(metrics["macro_f1"], 4))
    print("Critical recall mean:", round(metrics["critical_event_recall_mean"], 4))
    print("Saved: models/svm.joblib")


if __name__ == "__main__":
    main()
