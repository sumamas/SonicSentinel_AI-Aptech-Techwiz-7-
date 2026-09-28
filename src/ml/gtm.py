"""Google Teachable Machine (TM audio / speech-commands) model, run in Python.

The exported TF.js files in gtm_model/ (model.json, weights.bin, metadata.json)
are executed with plain NumPy, so no TensorFlow.js or browser is needed and the
SAME audio segment used by the Python model can be classified independently.

Feature extraction reproduces the browser pipeline used by Teachable Machine
(speech-commands BrowserFftFeatureExtractor):
  44.1 kHz audio -> 1024-point frames (hop 1024) -> Blackman window ->
  |FFT|/N in dB (Web Audio AnalyserNode, smoothing 0) -> first 232 bins ->
  43 frames (~1 s) -> per-example z-normalisation -> CNN -> softmax.

The Python prediction is never passed to this model (SRS 1.8 rule 14).
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GTM_DIR = ROOT / "gtm_model"
GTM_SAMPLE_RATE = 44100
FFT_SIZE = 1024
FREQ_BINS = 232
FRAMES = 43
WINDOW_SAMPLES = FFT_SIZE * FRAMES          # 44 032 samples ~ 0.998 s
GTM_PROJECT_URL = "https://teachablemachine.withgoogle.com/models/wziUFlrWa/"

# Canonical SRS class names. GTM export labels can contain stray spaces or
# different capitalisation ("Gunshot  ", "Glass Breaking "); they are mapped
# by comparing a normalised key only. No probabilities are changed.
_CANONICAL = [
    "Machinery Fault", "Glass Breaking", "Alarm or Siren", "Vehicle Horn",
    "Animal Sound", "Gunshot", "Panic Scream", "Aggression",
    "Person Asking for Help", "Background Noise",
]


def _key(text: str) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


_ALIASES = {_key(c): c for c in _CANONICAL}
_ALIASES.update({
    _key("Alarm/Siren"): "Alarm or Siren", _key("Siren"): "Alarm or Siren",
    _key("Aggression or Violent Conflict"): "Aggression",
    _key("_background_noise_"): "Background Noise", _key("Background"): "Background Noise",
    _key("Help"): "Person Asking for Help",
})


def canonical_label(label: str) -> str:
    return _ALIASES.get(_key(label), str(label).strip())


# --------------------------------------------------------------------- model
class GTMModel:
    def __init__(self, folder: Path = GTM_DIR):
        folder = Path(folder)
        model_path, weights_path, meta_path = folder / "model.json", folder / "weights.bin", folder / "metadata.json"
        for p in (model_path, weights_path, meta_path):
            if not p.is_file():
                raise FileNotFoundError(f"GTM export file missing: {p}. Copy model.json, weights.bin and metadata.json into gtm_model/.")
        self.model_json = json.loads(model_path.read_text(encoding="utf-8"))
        self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        raw = np.frombuffer(weights_path.read_bytes(), dtype="<f4")
        self.weights: dict[str, np.ndarray] = {}
        offset = 0
        for group in self.model_json["weightsManifest"]:
            for spec in group["weights"]:
                if spec.get("dtype", "float32") != "float32":
                    raise ValueError("Only float32 GTM weights are supported.")
                size = int(np.prod(spec["shape"]))
                self.weights[spec["name"]] = raw[offset:offset + size].reshape(spec["shape"]).astype(np.float32)
                offset += size
        if offset != raw.size:
            raise ValueError("weights.bin size does not match model.json manifest.")
        self.raw_labels = [str(x) for x in self.metadata.get("wordLabels", [])]
        self.labels = [canonical_label(x) for x in self.raw_labels]
        self.layers = self._layers(self.model_json["modelTopology"])
        head = [l for l in self.layers if l["class_name"] == "Dense"][-1]
        if self.weights[head["name"] + "/kernel"].shape[1] != len(self.labels):
            raise ValueError("GTM output count and metadata labels disagree.")
        digest = hashlib.sha256()
        for p in (model_path, weights_path, meta_path):
            digest.update(p.read_bytes())
        self.sha256 = digest.hexdigest()
        self.version = f"{self.metadata.get('modelName', 'TM')}@{self.metadata.get('timeStamp', 'unknown')}"

    def _layers(self, topology):
        flat = []

        def walk(node):
            if node.get("class_name") in ("Model", "Sequential", "Functional"):
                cfg = node["config"]
                for layer in (cfg["layers"] if isinstance(cfg, dict) else cfg):
                    walk(layer)
            else:
                cfg = node.get("config", {})
                flat.append({"class_name": node["class_name"], "name": cfg.get("name", node.get("name")), "config": cfg})
        walk(topology)
        return flat

    # -- tiny NumPy implementations of the layers used by TM audio ----------
    @staticmethod
    def _conv2d_valid(x, kernel, bias):
        # x: (N,H,W,C)  kernel: (kh,kw,C,F)
        kh, kw, _, filters = kernel.shape
        n, h, w, c = x.shape
        oh, ow = h - kh + 1, w - kw + 1
        patches = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(1, 2))  # N,oh,ow,C,kh,kw
        patches = patches.transpose(0, 1, 2, 4, 5, 3).reshape(n, oh, ow, kh * kw * c)
        return patches @ kernel.reshape(kh * kw * c, filters) + bias

    @staticmethod
    def _maxpool(x, ph, pw, sh, sw):
        win = np.lib.stride_tricks.sliding_window_view(x, (ph, pw), axis=(1, 2))  # N,H',W',C,ph,pw
        return win[:, ::sh, ::sw].max(axis=(4, 5))

    def forward(self, batch: np.ndarray) -> np.ndarray:
        x = batch.astype(np.float32)
        for layer in self.layers:
            kind, cfg, name = layer["class_name"], layer["config"], layer["name"]
            if kind == "InputLayer" or kind == "Dropout":
                continue
            if kind == "Conv2D":
                if cfg.get("padding", "valid") != "valid" or tuple(cfg.get("strides", [1, 1])) != (1, 1):
                    raise ValueError("Unsupported Conv2D configuration in GTM model.")
                x = self._conv2d_valid(x, self.weights[name + "/kernel"], self.weights[name + "/bias"])
            elif kind == "MaxPooling2D":
                ph, pw = cfg.get("pool_size", [2, 2])
                sh, sw = cfg.get("strides") or [ph, pw]
                if cfg.get("padding", "valid") != "valid":
                    raise ValueError("Unsupported pooling padding in GTM model.")
                x = self._maxpool(x, ph, pw, sh, sw)
            elif kind == "Flatten":
                x = x.reshape(x.shape[0], -1)
            elif kind == "Dense":
                x = x @ self.weights[name + "/kernel"] + self.weights[name + "/bias"]
            else:
                raise ValueError(f"Unsupported GTM layer: {kind}")
            act = cfg.get("activation")
            if act == "relu":
                x = np.maximum(x, 0)
            elif act == "softmax":
                x = x - x.max(axis=1, keepdims=True)
                x = np.exp(x)
                x = x / x.sum(axis=1, keepdims=True)
            elif act not in (None, "linear"):
                raise ValueError(f"Unsupported activation {act}")
        return x


@lru_cache(maxsize=1)
def _cached_model(token):
    return GTMModel(GTM_DIR)


def load_gtm() -> GTMModel:
    files = [GTM_DIR / n for n in ("model.json", "weights.bin", "metadata.json")]
    token = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files if p.is_file())
    return _cached_model(token)


def gtm_available() -> bool:
    try:
        load_gtm()
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ features
_BLACKMAN = (0.42 - 0.5 * np.cos(2 * np.pi * np.arange(FFT_SIZE) / FFT_SIZE)
             + 0.08 * np.cos(4 * np.pi * np.arange(FFT_SIZE) / FFT_SIZE)).astype(np.float64)


def to_gtm_rate(y: np.ndarray, sr: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if sr != GTM_SAMPLE_RATE:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=GTM_SAMPLE_RATE)
    return y.astype(np.float32)


def spectrogram_frames(y44: np.ndarray) -> np.ndarray:
    """dB spectrum frames exactly like AnalyserNode.getFloatFrequencyData."""
    n = len(y44) // FFT_SIZE
    if n == 0:
        y44 = np.pad(y44, (0, FFT_SIZE - len(y44)))
        n = 1
    frames = y44[:n * FFT_SIZE].reshape(n, FFT_SIZE).astype(np.float64) * _BLACKMAN
    mag = np.abs(np.fft.rfft(frames, axis=1))[:, :FREQ_BINS] / FFT_SIZE
    # Real microphones never deliver digital zero; floor at -100 dB (AnalyserNode
    # default minDecibels) so zero padding cannot dominate the z-normalisation.
    return (20.0 * np.log10(np.maximum(mag, 1e-5))).astype(np.float32)


def _normalise(x):
    mean, std = float(x.mean()), float(x.std())
    return (x - mean) / (std + 1e-6)


def windows_for_segment(y44: np.ndarray, hop_frames: int = 11):
    """~1 s GTM windows sliding over the segment (hop ~0.26 s)."""
    spec = spectrogram_frames(y44)
    if spec.shape[0] < FRAMES:
        pad = np.full((FRAMES - spec.shape[0], FREQ_BINS), spec.min() if spec.size else -120, dtype=np.float32)
        spec = np.concatenate([spec, pad])
    starts = list(range(0, spec.shape[0] - FRAMES + 1, hop_frames))
    if starts[-1] != spec.shape[0] - FRAMES:
        starts.append(spec.shape[0] - FRAMES)
    wins, energy = [], []
    for s in starts:
        seg = y44[s * FFT_SIZE:(s + FRAMES) * FFT_SIZE]
        energy.append(float(np.sqrt(np.mean(np.square(seg)))) if seg.size else 0.0)
        wins.append(_normalise(spec[s:s + FRAMES]))
    return np.stack(wins)[..., None], np.asarray(energy), [s * FFT_SIZE / GTM_SAMPLE_RATE for s in starts]


def classify_signal(y: np.ndarray, sr: int) -> dict:
    """Classify one audio segment with the GTM model. Returns scores for every class."""
    model = load_gtm()
    y = np.asarray(y, dtype=np.float32)
    nz = np.flatnonzero(np.abs(y) > 1e-7)
    if nz.size:                                   # drop zero padding added by fixed-length preprocessing
        y = y[nz[0]:nz[-1] + 1]
    y44 = to_gtm_rate(y, sr)
    if not y44.size or not np.all(np.isfinite(y44)):
        raise ValueError("GTM received an empty or invalid audio segment.")
    x, energy, starts = windows_for_segment(y44)
    probs = model.forward(x)                       # (windows, classes)
    # Aggregate windows that actually contain the sound (>= 50 % of the loudest
    # window energy). Quiet padding would otherwise vote for Background Noise.
    active = energy >= 0.5 * energy.max() if energy.max() > 0 else np.ones_like(energy, bool)
    weights = energy * active
    weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(energy), 1 / len(energy))
    p = (probs * weights[:, None]).sum(axis=0)
    p = p / p.sum()
    scores = {}
    for label, value in zip(model.labels, p):
        scores[label] = scores.get(label, 0.0) + float(value)
    for label in _CANONICAL:
        scores.setdefault(label, 0.0)
    order = sorted(scores, key=lambda k: -scores[k])
    top = [{"label": k, "confidence": scores[k]} for k in order[:3]]
    return {
        "status": "ready",
        "model": "google_teachable_machine",
        "model_version": model.version,
        "model_sha256": model.sha256,
        "project_url": GTM_PROJECT_URL,
        "prediction": top[0],
        "top3": top,
        "all_confidences": scores,
        "top_two_margin": float(top[0]["confidence"] - top[1]["confidence"]),
        "windows_used": int(active.sum()),
        "windows_total": int(len(energy)),
        "input_policy": "same selected segment as the Python model; ~1 s TM windows (44.1 kHz FFT features), energy-weighted average",
    }


def classify_file(path) -> dict:
    from src.audio.decode import decode_mono
    y, sr = decode_mono(path)
    return classify_signal(y, sr)
