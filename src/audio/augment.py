from __future__ import annotations

import numpy as np
import librosa

from .preprocess import fix_length


def add_noise_at_snr(y: np.ndarray, rng: np.random.Generator, snr_db: float) -> np.ndarray:
    signal_power = float(np.mean(y ** 2)) + 1e-12
    noise = rng.normal(0.0, 1.0, size=y.shape).astype(np.float32)
    noise_power = float(np.mean(noise ** 2)) + 1e-12
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise *= np.sqrt(target_noise_power / noise_power)
    return (y + noise).astype(np.float32)


def random_gain(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gain_db = float(rng.uniform(-6.0, 6.0))
    gain = 10.0 ** (gain_db / 20.0)
    return (y * gain).astype(np.float32)


def random_shift(y: np.ndarray, rng: np.random.Generator, max_fraction: float = 0.18) -> np.ndarray:
    max_shift = max(1, int(len(y) * max_fraction))
    shift = int(rng.integers(-max_shift, max_shift + 1))
    return np.roll(y, shift).astype(np.float32)


def random_pitch(y: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    semitones = float(rng.uniform(-2.0, 2.0))
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones).astype(np.float32)


def random_time_stretch(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rate = float(rng.uniform(0.90, 1.10))
    stretched = librosa.effects.time_stretch(y, rate=rate)
    return fix_length(stretched, len(y))


def augment_signal(
    y: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    strength: str = "normal",
) -> np.ndarray:
    """Create one in-memory training augmentation.

    Validation and test audio must never call this function.
    No augmented file is counted as an original dataset recording.
    """
    out = np.array(y, dtype=np.float32, copy=True)

    operations = ["noise", "gain", "shift", "pitch", "stretch"]
    count = 2 if strength == "normal" else 1
    selected = rng.choice(operations, size=count, replace=False)

    for operation in selected:
        if operation == "noise":
            out = add_noise_at_snr(out, rng, snr_db=float(rng.uniform(12.0, 28.0)))
        elif operation == "gain":
            out = random_gain(out, rng)
        elif operation == "shift":
            out = random_shift(out, rng)
        elif operation == "pitch":
            out = random_pitch(out, sr, rng)
        elif operation == "stretch":
            out = random_time_stretch(out, rng)

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak
    return fix_length(out, len(y)).astype(np.float32)
