import numpy as np

from src.audio.features import extract_features_from_signal, log_mel_from_signal
from src.ml.training_config import SAMPLE_RATE, NUM_SAMPLES, N_MELS, TARGET_MEL_FRAMES


def test_feature_vector_is_finite():
    t = np.arange(NUM_SAMPLES) / SAMPLE_RATE
    signal = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    vector = extract_features_from_signal(signal, SAMPLE_RATE)
    assert vector.ndim == 1
    assert vector.size > 100
    assert np.all(np.isfinite(vector))


def test_log_mel_shape():
    t = np.arange(NUM_SAMPLES) / SAMPLE_RATE
    signal = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    mel = log_mel_from_signal(signal, SAMPLE_RATE)
    assert mel.shape == (N_MELS, TARGET_MEL_FRAMES)
    assert np.all(np.isfinite(mel))
