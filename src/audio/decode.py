"""One canonical decoder for preparation, training and new-file inference."""
from pathlib import Path
from functools import lru_cache
import shutil
import subprocess
import numpy as np
import soundfile as sf

SAMPLE_RATE = 22050


@lru_cache(maxsize=1)
def ffmpeg_executable():
    found = shutil.which('ffmpeg')
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError('FFmpeg missing. Install this export\'s requirements-windows-ml.txt.') from exc


def decode_mono(path, max_seconds=61.0):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    # Prepared WAV files are already decoded by this exact pipeline.
    if path.suffix.lower() == '.wav':
        try:
            info = sf.info(path)
            if info.channels == 1 and info.samplerate == SAMPLE_RATE:
                y, _ = sf.read(path, dtype='float32', frames=int(max_seconds * SAMPLE_RATE))
                if y.size and np.all(np.isfinite(y)):
                    return y, SAMPLE_RATE
        except (RuntimeError, OSError):
            pass
    result = subprocess.run([
        ffmpeg_executable(), '-nostdin', '-v', 'error', '-threads', '1',
        '-i', str(path), '-map', '0:a:0', '-t', str(max_seconds),
        '-ac', '1', '-ar', str(SAMPLE_RATE), '-c:a', 'pcm_f32le', '-f', 'f32le', 'pipe:1'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if result.returncode:
        raise ValueError(result.stderr.decode('utf-8', 'replace')[-1200:])
    y = np.frombuffer(result.stdout, dtype='<f4').copy()
    if not y.size or not np.all(np.isfinite(y)):
        raise ValueError('Empty or non-finite decoded audio.')
    return y, SAMPLE_RATE
