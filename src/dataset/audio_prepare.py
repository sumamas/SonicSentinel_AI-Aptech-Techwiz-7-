from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf

from .starter_config import (
    CLIPPING_AMPLITUDE_THRESHOLD,
    SILENCE_AMPLITUDE_THRESHOLD,
    TARGET_DURATION_SECONDS,
    TARGET_SAMPLE_RATE,
    TARGET_SAMPLES,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_audio_info(path: Path) -> Tuple[float, int, int]:
    """Return (duration_seconds, sample_rate, channels)."""
    try:
        info = sf.info(str(path))
        if info.samplerate <= 0:
            raise ValueError("Invalid sample rate")
        return float(info.frames / info.samplerate), int(info.samplerate), int(info.channels)
    except Exception:
        # Fallback for codecs not handled by the local libsndfile build.
        y, sr = librosa.load(str(path), sr=None, mono=False)
        channels = 1 if y.ndim == 1 else int(y.shape[0])
        samples = y.shape[-1]
        return float(samples / sr), int(sr), channels


def load_mono_resampled(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=TARGET_SAMPLE_RATE, mono=True)
    y = np.asarray(y, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


def peak_center_crop_or_pad(y: np.ndarray) -> np.ndarray:
    """
    Produce exactly 3 seconds.

    * short clips are symmetrically zero-padded;
    * clips >3 seconds are cropped around the strongest RMS-energy region,
      rather than blindly taking only the first 3 seconds.
    """
    if len(y) == TARGET_SAMPLES:
        return y

    if len(y) < TARGET_SAMPLES:
        missing = TARGET_SAMPLES - len(y)
        left = missing // 2
        right = missing - left
        return np.pad(y, (left, right), mode="constant")

    # Find peak RMS frame and center a 3-second window around it.
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    peak_frame = int(np.argmax(rms)) if len(rms) else 0
    peak_sample = peak_frame * 512
    start = max(0, peak_sample - TARGET_SAMPLES // 2)
    start = min(start, len(y) - TARGET_SAMPLES)
    return y[start : start + TARGET_SAMPLES]


def normalize_peak(y: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak <= 1e-8:
        return y.astype(np.float32)
    scale = min(target_peak / peak, 10.0)
    return np.clip(y * scale, -1.0, 1.0).astype(np.float32)


def prepare_audio(path: Path) -> np.ndarray:
    y = load_mono_resampled(path)
    y = peak_center_crop_or_pad(y)
    y = normalize_peak(y)
    if len(y) != TARGET_SAMPLES:
        raise RuntimeError(f"Prepared audio has {len(y)} samples, expected {TARGET_SAMPLES}")
    return y


def quality_metrics(y: np.ndarray) -> dict:
    if len(y) == 0:
        return {
            "rms": 0.0,
            "silence_ratio": 1.0,
            "clipping_ratio": 0.0,
        }
    abs_y = np.abs(y)
    return {
        "rms": round(float(np.sqrt(np.mean(np.square(y)))), 6),
        "silence_ratio": round(float(np.mean(abs_y < SILENCE_AMPLITUDE_THRESHOLD)), 6),
        "clipping_ratio": round(float(np.mean(abs_y >= CLIPPING_AMPLITUDE_THRESHOLD)), 6),
    }


def write_wav(path: Path, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, TARGET_SAMPLE_RATE, subtype="PCM_16")
