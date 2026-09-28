from pathlib import Path

# Starter prototype only. Final SonicSentinel dataset must be expanded to all 10 SRS classes.
ACTIVE_CLASSES = {
    "alarm_siren": "Alarm or Siren",
    "glass_breaking": "Glass Breaking",
    "gunshot": "Gunshot",
}

TARGET_SAMPLE_RATE = 22_050
TARGET_DURATION_SECONDS = 3.0
TARGET_SAMPLES = int(TARGET_SAMPLE_RATE * TARGET_DURATION_SECONDS)

# The uploaded starter dataset contains one 360-second gunshot file.
# For the small prototype it is excluded so that a single recording cannot dominate training.
# Later it can be segmented, but all segments must retain the same original Audio ID/split.
STARTER_MAX_ORIGINAL_DURATION_SECONDS = 10.0

SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

RANDOM_SEED = 42
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Simple quality-inspection thresholds used only for metadata/reporting.
SILENCE_AMPLITUDE_THRESHOLD = 0.01
CLIPPING_AMPLITUDE_THRESHOLD = 0.99
