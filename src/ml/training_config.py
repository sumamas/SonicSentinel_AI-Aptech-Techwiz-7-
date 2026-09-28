from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
FEATURE_DIR = DATA_DIR / "features"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "model_evaluation"

SEED = 42

SAMPLE_RATE = 22_050
DURATION_SECONDS = 3.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SECONDS)

N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
TARGET_MEL_FRAMES = 130

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

ALLOWED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
    ".opus",
}

# New 10-class dataset version; old nine-class weights are not compatible.
CLASSES = ['Machinery Fault', 'Glass Breaking', 'Alarm or Siren', 'Vehicle Horn', 'Animal Sound', 'Gunshot', 'Panic Scream', 'Aggression', 'Person Asking for Help', 'Background Noise']

FOLDER_TO_LABEL = {'machinery_fault': 'Machinery Fault', 'glass_breaking': 'Glass Breaking', 'alarm_siren': 'Alarm or Siren', 'vehicle_horn': 'Vehicle Horn', 'animal_sound': 'Animal Sound', 'gunshot': 'Gunshot', 'panic_scream': 'Panic Scream', 'aggression': 'Aggression', 'person_asking_for_help': 'Person Asking for Help', 'background_noise': 'Background Noise'}

LABEL_TO_FOLDER = {
    label: folder
    for folder, label in FOLDER_TO_LABEL.items()
}

CLASS_TO_INDEX = {
    label: index
    for index, label in enumerate(CLASSES)
}

INDEX_TO_CLASS = {
    index: label
    for label, index in CLASS_TO_INDEX.items()
}

CRITICAL_CLASSES = {'Aggression', 'Glass Breaking', 'Gunshot', 'Panic Scream', 'Person Asking for Help'}

for directory in (
    SPLITS_DIR,
    FEATURE_DIR,
    MODELS_DIR,
    REPORTS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)