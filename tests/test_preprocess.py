import numpy as np
from src.audio.quality import assess_quality


def test_quality_unusable_silence():
    y = np.zeros(22050, dtype=np.float32)
    assert assess_quality(y)['label'] == 'Unusable'
