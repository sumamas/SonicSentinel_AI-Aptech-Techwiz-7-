from __future__ import annotations

import numpy as np
import librosa
from src.audio.decode import decode_mono

from src.ml.training_config import SAMPLE_RATE, DURATION_SECONDS, NUM_SAMPLES


def _safe_peak_normalize(y: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-8:
        y = y / peak
    return y.astype(np.float32)


def fix_length(y: np.ndarray, target_samples: int = NUM_SAMPLES) -> np.ndarray:
    """Pad or truncate an audio signal to a deterministic model input length."""
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    else:
        y = y[:target_samples]
    return y.astype(np.float32)


def strongest_window_start(y: np.ndarray, window: int, hop: int = 512) -> int:
    """Highest-energy full window; ties use the earliest start.

    Energy locates a candidate, not a semantic event. Labels still need review.
    Selection is identical for training, validation, test and file inference.
    """
    if len(y) <= window:
        return 0
    starts = np.unique(np.append(np.arange(0, len(y) - window + 1, hop), len(y) - window))
    power = np.concatenate(([0.0], np.cumsum(np.square(y.astype(np.float64)))))
    energies = power[starts + window] - power[starts]
    return int(starts[int(np.argmax(energies))])


def preprocess_signal(
    y: np.ndarray,
    sr: int,
    target_sr: int = SAMPLE_RATE,
    duration: float = DURATION_SECONDS,
    trim_silence: bool = True,
) -> tuple[np.ndarray, int]:
    """Standard SonicSentinel preprocessing used by training and inference.

    Steps: mono -> resample -> silence trim -> highest-energy full window ->
    peak normalization -> fixed-duration padding. This is a new v5 contract.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = librosa.to_mono(y)
    if y.size == 0:
        raise ValueError("Audio contains no usable signal.")

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if trim_silence and y.size:
        trimmed, _ = librosa.effects.trim(y, top_db=35)
        # Do not turn an almost-silent recording into an empty array.
        if trimmed.size:
            y = trimmed

    target_samples = int(target_sr * duration)
    start = strongest_window_start(y, target_samples)
    y = y[start:start + target_samples]
    y = _safe_peak_normalize(y)
    y = fix_length(y, target_samples)
    return y, target_sr


def load_audio(
    file_path: str,
    sr: int = SAMPLE_RATE,
    duration: float = DURATION_SECONDS,
) -> tuple[np.ndarray, int]:
    y, native_sr = decode_mono(file_path)
    return preprocess_signal(y, native_sr, target_sr=sr, duration=duration)


def load_audio_native(file_path: str) -> tuple[np.ndarray, int]:
    """Load without trimming/padding; useful for validation and segmentation."""
    y, sr = decode_mono(file_path)
    if y.size == 0:
        raise ValueError("Audio contains no usable signal.")
    return y.astype(np.float32), int(sr)


def segment_audio(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    window_seconds: float = DURATION_SECONDS,
    hop_seconds: float | None = None,
) -> list[np.ndarray]:
    """Split a signal into fixed windows. Derived windows always inherit the
    original recording's dataset split; the split is never decided here.
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if hop_seconds is None:
        hop_seconds = window_seconds
    if hop_seconds <= 0:
        raise ValueError("hop_seconds must be positive")

    window = int(sr * window_seconds)
    hop = int(sr * hop_seconds)
    segments: list[np.ndarray] = []
    for start in range(0, max(len(y), 1), hop):
        segment = y[start:start + window]
        if segment.size == 0:
            break
        segments.append(fix_length(segment, window))
        if start + window >= len(y):
            break
    return segments
