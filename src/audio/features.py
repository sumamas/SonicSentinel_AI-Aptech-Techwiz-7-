from __future__ import annotations

import numpy as np
import librosa

from .preprocess import load_audio
from src.ml.training_config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    TARGET_MEL_FRAMES,
)


def _mean_std(feature: np.ndarray) -> list[float]:
    feature = np.asarray(feature)
    if feature.ndim == 1:
        return [float(np.mean(feature)), float(np.std(feature))]
    return np.concatenate([
        np.mean(feature, axis=1),
        np.std(feature, axis=1),
    ]).astype(np.float32).tolist()


def extract_features_from_signal(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Feature vector for classical models (Random Forest / SVM)."""
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=40, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=64, n_fft=N_FFT, hop_length=HOP_LENGTH, power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)

    # librosa.beat.beat_track can return an ndarray on some versions.
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    tempo_value = float(np.ravel(tempo)[0]) if np.size(tempo) else 0.0

    values: list[float] = []
    values.extend(_mean_std(mfcc))
    values.extend(_mean_std(log_mel))
    values.extend(_mean_std(chroma))
    values.extend(_mean_std(zcr))
    values.extend(_mean_std(rms))
    values.extend(_mean_std(centroid))
    values.extend(_mean_std(bandwidth))
    values.extend(_mean_std(rolloff))
    values.extend(_mean_std(onset))
    values.append(tempo_value)

    vector = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(vector)):
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    return vector


def extract_features(file_path: str) -> np.ndarray:
    y, sr = load_audio(file_path)
    return extract_features_from_signal(y, sr)


def log_mel_from_signal(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    target_frames: int = TARGET_MEL_FRAMES,
    normalize: bool = True,
) -> np.ndarray:
    """Create the 2-D Log-Mel image consumed by the custom CNN."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=n_mels,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    if log_mel.shape[1] < target_frames:
        pad = target_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="constant", constant_values=-80.0)
    else:
        log_mel = log_mel[:, :target_frames]

    if normalize:
        mean = float(np.mean(log_mel))
        std = float(np.std(log_mel))
        log_mel = (log_mel - mean) / (std + 1e-6)

    return log_mel.astype(np.float32)


def log_mel_spectrogram(file_path: str) -> np.ndarray:
    y, sr = load_audio(file_path)
    return log_mel_from_signal(y, sr)


def mel_spectrogram(file_path: str, n_mels: int = N_MELS) -> np.ndarray:
    """Backward-compatible alias used by older web code."""
    y, sr = load_audio(file_path)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)
