from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import json
import hashlib
from importlib.metadata import version
from src.ml.reliability import sha256_file

from src.audio.features import extract_features, extract_features_from_signal, log_mel_from_signal
from src.audio.preprocess import load_audio
from src.audio.augment import augment_signal
from src.ml.training_config import ROOT, SPLITS_DIR, FEATURE_DIR, CLASS_TO_INDEX, SEED


def read_split(split: str, root: str | Path = ROOT) -> pd.DataFrame:
    root = Path(root)
    path = root / "data" / "splits" / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Use a complete prepared ten-class dataset ZIP; do not regenerate frozen splits."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"Split {split!r} is empty.")
    df["relative_path"] = df["relative_path"].astype(str).str.replace("\\", "/", regex=False)
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

    if failed:
        raise RuntimeError(f"Feature extraction failed for {len(failed)} recordings: {failed[:5]}")
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
    out.with_suffix(".json").write_text(json.dumps({"fingerprint": cache_fingerprint(split)}, indent=2), encoding="utf-8")
    return out


def load_feature_cache(split: str, feature_dir: str | Path = FEATURE_DIR):
    path = Path(feature_dir) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature cache {path} not found. Run scripts/build_feature_cache.py first."
        )
    manifest = path.with_suffix(".json")
    if not manifest.exists() or json.loads(manifest.read_text())["fingerprint"] != cache_fingerprint(split):
        raise RuntimeError("Feature cache is stale or unverified. Re-run scripts/build_feature_cache.py in the training workspace.")
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


def cache_fingerprint(split):
    digest = hashlib.sha256()
    for package in ("numpy", "librosa", "scipy", "soundfile"):
        digest.update(f"{package}=={version(package)}".encode())
    for relative in (f"data/splits/{split}.csv", "src/ml/training_config.py", "src/audio/features.py", "src/audio/preprocess.py", "src/audio/decode.py", "src/audio/augment.py", "src/ml/dataset.py"):
        digest.update(sha256_file(ROOT / relative).encode())
    for _, row in read_split(split).iterrows():
        actual = sha256_file(ROOT / row["relative_path"])
        expected = str(row.get("sha256", ""))
        if expected and expected != "nan" and expected != actual:
            raise RuntimeError(f"Audio changed after splitting: {row['relative_path']}")
        digest.update(actual.encode())
    return digest.hexdigest()
