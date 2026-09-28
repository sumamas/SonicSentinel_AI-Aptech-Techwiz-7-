from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd

from src.audio.preprocess import load_audio
from src.audio.augment import augment_signal
from src.audio.features import log_mel_from_signal
from src.ml.training_config import ROOT, CLASS_TO_INDEX, N_MELS, TARGET_MEL_FRAMES, SEED


def _require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for the custom CNN. Install it with: "
            "python -m pip install -r requirements-ml.txt"
        ) from exc
    return tf


class AudioSequence:
    """Small wrapper that builds a tf.keras Sequence lazily.

    Defining it without importing TensorFlow at module import time keeps the
    rest of SonicSentinel usable before the CNN dependency is installed.
    """

    def __new__(
        cls,
        dataframe: pd.DataFrame,
        batch_size: int = 16,
        shuffle: bool = True,
        augment: bool = False,
        seed: int = SEED,
    ):
        tf = _require_tensorflow()

        class _Sequence(tf.keras.utils.Sequence):
            def __init__(self):
                super().__init__(workers=2, use_multiprocessing=False, max_queue_size=2)
                self.df = dataframe.reset_index(drop=True).copy()
                self.batch_size = batch_size
                self.shuffle = shuffle
                self.augment = augment
                self.rng = np.random.default_rng(seed)
                self.indices = np.arange(len(self.df))
                self.on_epoch_end()

            def __len__(self):
                return math.ceil(len(self.df) / self.batch_size)

            def on_epoch_end(self):
                if self.shuffle:
                    self.rng.shuffle(self.indices)

            def __getitem__(self, index):
                batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
                X = np.zeros((len(batch_indices), N_MELS, TARGET_MEL_FRAMES, 1), dtype=np.float32)
                y = np.zeros((len(batch_indices),), dtype=np.int32)

                for out_index, row_index in enumerate(batch_indices):
                    row = self.df.iloc[int(row_index)]
                    path = ROOT / str(row["relative_path"])
                    signal, sr = load_audio(str(path))
                    if self.augment:
                        signal = augment_signal(signal, sr, self.rng)
                    X[out_index, ..., 0] = log_mel_from_signal(signal, sr)
                    y[out_index] = CLASS_TO_INDEX[str(row["class_label"])]
                return X, y

        return _Sequence()
