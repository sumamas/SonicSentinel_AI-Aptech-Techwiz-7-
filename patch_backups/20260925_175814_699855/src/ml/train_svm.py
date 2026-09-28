from __future__ import annotations

import argparse
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.ml.dataset import load_feature_cache
from src.ml.evaluation import metrics_from_predictions, save_confusion_matrix, save_metrics
from src.ml.training_config import CLASSES, MODELS_DIR, REPORTS_DIR, SEED


def main():
    parser = argparse.ArgumentParser(description="Train SonicSentinel SVM baseline.")
    parser.add_argument("--quick", action="store_true", help="Use a smaller tuning grid for development runs.")
    args = parser.parse_args()

    X_train, y_train, train_ids = load_feature_cache("train")
    X_val, y_val, _ = load_feature_cache("validation")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)),
    ])
    grid = {
        "svc__C": [3, 10] if args.quick else [1, 3, 10, 30],
        "svc__gamma": ["scale", 0.01] if args.quick else ["scale", 0.01, 0.001],
    }
    groups = np.asarray([sample_id.split("__aug")[0] for sample_id in train_ids])
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipeline, grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
    search.fit(X_train, y_train, groups=groups)

    model = search.best_estimator_
    val_pred = model.predict(X_val)
    metrics = metrics_from_predictions(y_val, val_pred, "SVM")
    metrics["best_params"] = search.best_params_
    metrics["cv_best_macro_f1"] = float(search.best_score_)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_name": "svm",
        "display_name": "SVM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classes": CLASSES,
        "feature_count": int(X_train.shape[1]),
        "best_params": search.best_params_,
        "model": model,
    }
    joblib.dump(bundle, MODELS_DIR / "svm.joblib")

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
