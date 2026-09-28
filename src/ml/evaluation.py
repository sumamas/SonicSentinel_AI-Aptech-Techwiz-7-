from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.ml.training_config import CLASSES, CRITICAL_CLASSES


def metrics_from_predictions(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    model_name: str,
) -> dict:
    y_true = np.asarray(list(y_true), dtype=str)
    y_pred = np.asarray(list(y_pred), dtype=str)

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASSES,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    critical_recalls = [
        float(report[label]["recall"])
        for label in CLASSES
        if label in CRITICAL_CLASSES and label in report
    ]

    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "critical_event_recall_mean": float(np.mean(critical_recalls)) if critical_recalls else 0.0,
        "class_report": report,
    }


def save_metrics(metrics: dict, output_dir: Path, stem: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}_metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    rows = []
    for label in CLASSES:
        values = metrics.get("class_report", {}).get(label, {})
        rows.append({
            "class": label,
            "precision": values.get("precision", 0.0),
            "recall": values.get("recall", 0.0),
            "f1_score": values.get("f1-score", 0.0),
            "support": values.get("support", 0.0),
        })
    pd.DataFrame(rows).to_csv(output_dir / f"{stem}_class_metrics.csv", index=False)
    return path


def save_confusion_matrix(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    output_path: Path,
    title: str,
) -> None:
    y_true = np.asarray(list(y_true), dtype=str)
    y_pred = np.asarray(list(y_pred), dtype=str)
    matrix = confusion_matrix(y_true, y_pred, labels=CLASSES)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path.with_suffix(".csv"), matrix, delimiter=",", fmt="%d")

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(CLASSES)),
        yticks=np.arange(len(CLASSES)),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2.0 if matrix.size and matrix.max() else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
