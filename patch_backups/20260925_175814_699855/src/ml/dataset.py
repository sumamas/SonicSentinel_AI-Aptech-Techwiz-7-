from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.audio.features import extract_features, extract_features_from_signal, log_mel_from_signal
from src.audio.preprocess import load_audio
from src.audio.augment import augment_signal
from src.ml.training_config import ROOT, SPLITS_DIR, FEATURE_DIR, CLASS_TO_INDEX, SEED


def read_split(split: str, root: str | Path = ROOT) -> pd.DataFrame:
    root = Path(root)
    path = root / "data" / "splits" / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/create_metadata.py and scripts/split_dataset.py first."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"Split {split!r} is empty.")
    return df


def resolve_audio_path(row: pd.Series, root: str | Path = ROOT) -> Path:
    root = Path(root)
    return root / str(row["relative_path"])


def load_feature_dataset(
    root: str | Path,
    split: str,
    augment_per_sample: int = 0,
    seed: int = SEED,
):
    """Extract classical ML features directly from split CSV files.

    Training augmentation is generated in-memory. Validation/test callers should
    leave augment_per_sample=0.
    """
    root = Path(root)
    df = read_split(split, root)
    rng = np.random.default_rng(seed)

    X: list[np.ndarray] = []
    y: list[str] = []
    ids: list[str] = []
    failed: list[tuple[str, str]] = []

    for _, row in df.iterrows():
        path = resolve_audio_path(row, root)
        try:
            signal, sr = load_audio(str(path))
            X.append(extract_features_from_signal(signal, sr))
            y.append(str(row["class_label"]))
            ids.append(str(row["audio_id"]))

            if split == "train" and augment_per_sample > 0:
                for aug_index in range(augment_per_sample):
                    augmented = augment_signal(signal, sr, rng)
                    X.append(extract_features_from_signal(augmented, sr))
                    y.append(str(row["class_label"]))
                    ids.append(f"{row['audio_id']}__aug{aug_index + 1}")
        except Exception as exc:
            failed.append((str(path), str(exc)))

    if not X:
        raise RuntimeError(f"No usable samples found for split={split}. Failed={failed[:5]}")

    return np.vstack(X), np.asarray(y), np.asarray(ids), failed


def save_feature_cache(
    split: str,
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    feature_dir: str | Path = FEATURE_DIR,
) -> Path:
    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)
    out = feature_dir / f"{split}.npz"
    np.savez_compressed(out, X=X.astype(np.float32), y=y.astype(str), ids=ids.astype(str))
    return out


def load_feature_cache(split: str, feature_dir: str | Path = FEATURE_DIR):
    path = Path(feature_dir) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature cache {path} not found. Run scripts/build_feature_cache.py first."
        )
    data = np.load(path, allow_pickle=False)
    return data["X"], data["y"].astype(str), data["ids"].astype(str)


def load_cnn_example(file_path: str | Path, augment: bool = False, rng=None):
    signal, sr = load_audio(str(file_path))
    if augment:
        if rng is None:
            rng = np.random.default_rng(SEED)
        signal = augment_signal(signal, sr, rng)
    mel = log_mel_from_signal(signal, sr)
    return mel[..., np.newaxis]


def label_index(label: str) -> int:
    try:
        return CLASS_TO_INDEX[label]
    except KeyError as exc:
        raise ValueError(f"Unknown SonicSentinel class label: {label}") from exc
