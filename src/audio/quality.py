import numpy as np


def assess_quality(y: np.ndarray) -> dict:
    if y.size == 0:
        return {'label': 'Unusable', 'silence_ratio': 1.0, 'clipping_ratio': 0.0, 'rms': 0.0}

    abs_y = np.abs(y)
    silence_ratio = float(np.mean(abs_y < 0.01))
    clipping_ratio = float(np.mean(abs_y >= 0.99))
    rms = float(np.sqrt(np.mean(np.square(y))))

    if rms < 0.005 or silence_ratio > 0.95:
        label = 'Unusable'
    elif clipping_ratio > 0.05 or silence_ratio > 0.75:
        label = 'Poor'
    elif clipping_ratio > 0.01 or silence_ratio > 0.5:
        label = 'Acceptable'
    else:
        label = 'Good'

    return {
        'label': label,
        'silence_ratio': round(silence_ratio, 4),
        'clipping_ratio': round(clipping_ratio, 4),
        'rms': round(rms, 6),
    }
