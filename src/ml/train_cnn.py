from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from src.ml.cnn_data import AudioSequence
from src.ml.cnn_model import build_custom_cnn
from src.ml.provenance import input_contract, training_evidence
from src.ml.dataset import read_split
from src.ml.reliability import sha256_file
from src.ml.evaluation import metrics_from_predictions, save_confusion_matrix, save_metrics
from src.ml.training_config import (
    CLASS_TO_INDEX,
    INDEX_TO_CLASS,
    CLASSES,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
)


def require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is not installed. Run: "
            "python -m pip install -r requirements-ml.txt"
        ) from exc
    return tf


def plot_history(history, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history.history.get("loss", []), label="Train loss")
    ax.plot(history.history.get("val_loss", []), label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Custom CNN Training History")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_callbacks(tf, best_path, last_path, history_path):
    """Save best and last model/optimizer states before best-weight restoration."""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_path), monitor="val_loss", mode="min",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4,
            min_lr=1e-6, verbose=1,
        ),
        # Capture the end-of-epoch model and updated optimizer learning rate.
        # This runs before EarlyStopping can restore older model weights.
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(last_path), save_best_only=False,
            save_freq="epoch", verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(history_path)),
    ]


def main():
    parser = argparse.ArgumentParser(description="Train SonicSentinel custom CNN from scratch.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-augmentation", action="store_true")
    args = parser.parse_args()

    tf = require_tensorflow()
    tf.keras.utils.set_random_seed(SEED)

    train_df = read_split("train")
    val_df = read_split("validation")
    if set(train_df["class_label"])!=set(CLASSES) or set(val_df["class_label"])!=set(CLASSES):
        raise ValueError("CNN requires all active classes in train and validation.")

    train_sequence = AudioSequence(
        train_df,
        batch_size=args.batch_size,
        shuffle=True,
        augment=not args.no_augmentation,
        seed=SEED,
    )
    val_sequence = AudioSequence(
        val_df,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
        seed=SEED,
    )

    y_train_index = np.asarray([CLASS_TO_INDEX[label] for label in train_df["class_label"]], dtype=int)
    present_classes = np.unique(y_train_index)
    weights = compute_class_weight(class_weight="balanced", classes=present_classes, y=y_train_index)
    class_weight = {int(index): float(weight) for index, weight in zip(present_classes, weights)}

    model = build_custom_cnn()
    model.summary()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = MODELS_DIR / "custom_cnn.keras"

    last_path = MODELS_DIR / "custom_cnn_last.keras"
    callbacks = make_callbacks(
        tf, best_path, last_path,
        REPORTS_DIR / "custom_cnn_training_history.csv",
    )

    backup = os.environ.get("SONIC_BACKUP_DIR")
    if backup:
        class EpochBackup(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                # These are recovery artifacts. An interrupted epoch is not
                # a completed model and has no final CNN manifest yet.
                destination = Path(backup) / "cnn_epoch_checkpoints"
                destination.mkdir(parents=True, exist_ok=True)
                for p in [best_path, last_path, REPORTS_DIR / "custom_cnn_training_history.csv"]:
                    if p.is_file():
                        temp = destination / (p.name + ".copying")
                        shutil.copyfile(p, temp); temp.replace(destination / p.name)
                (destination / "progress.json").write_text(json.dumps({"epoch_completed":epoch+1,"finalized":False}))
        callbacks.append(EpochBackup())

    history = model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # The last checkpoint was saved at epoch end. Do not overwrite it here:
    # EarlyStopping has now restored best weights but not the old optimizer.
    plot_history(history, REPORTS_DIR / "custom_cnn_training_history.png")

    best_model = tf.keras.models.load_model(best_path)
    probabilities = best_model.predict(val_sequence, verbose=1)
    pred_indices = np.argmax(probabilities, axis=1)
    y_pred = np.asarray([INDEX_TO_CLASS[int(index)] for index in pred_indices])
    y_true = val_df["class_label"].astype(str).to_numpy()

    metrics = metrics_from_predictions(y_true, y_pred, "Custom CNN")
    metrics["training_evidence"] = training_evidence(cnn=True)
    metrics["trained_from_scratch"] = True
    metrics["pretrained_weights_used"] = False
    metrics["epochs_requested"] = args.epochs
    metrics["epochs_completed"] = len(history.history.get("loss", []))
    metrics["best_epoch"] = int(np.argmin(history.history["val_loss"])) + 1
    metrics["model_sha256"] = sha256_file(best_path)
    metrics["last_checkpoint_sha256"] = sha256_file(last_path)
    metrics["last_checkpoint_semantics"] = "End-of-epoch model and optimizer; callback counters/RNG are not a full resume snapshot."
    # Fresh validation evaluation writes its own report later. Preserve the
    # training provenance in a separate file that the evaluator never replaces.
    (REPORTS_DIR / "custom_cnn_training_summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8",
    )
    save_metrics(metrics, REPORTS_DIR, "custom_cnn_validation")
    save_confusion_matrix(
        y_true,
        y_pred,
        REPORTS_DIR / "custom_cnn_validation_confusion_matrix.png",
        "Custom CNN - Validation Confusion Matrix",
    )

    manifest = {
        "model_name": "custom_cnn",
        "input_contract": input_contract(cnn=True),
        "model_sha256": sha256_file(best_path),
        "display_name": "Custom CNN",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classes": CLASSES,
        "class_to_index": CLASS_TO_INDEX,
        "trained_from_scratch": True,
        "pretrained_weights_used": False,
        "input": "128x130 normalized Log-Mel spectrogram, single channel",
        "sample_rate": 22050,
        "duration_seconds": 3.0,
    }
    (MODELS_DIR / "custom_cnn_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nCustom CNN training complete")
    print("Validation accuracy:", round(metrics["accuracy"], 4))
    print("Validation macro F1:", round(metrics["macro_f1"], 4))
    print("Critical recall mean:", round(metrics["critical_event_recall_mean"], 4))
    print("Best model: models/custom_cnn.keras")
    print("Pretrained weights used: NO")


if __name__ == "__main__":
    main()
