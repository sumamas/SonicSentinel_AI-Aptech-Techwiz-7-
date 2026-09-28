# SonicSentinel — Starter 3-Class Dataset

This starter pipeline is intentionally limited to the audio currently available in the user's `raw.zip`:

- Alarm or Siren
- Glass Breaking
- Gunshot

It is **not** the final 10-class SRS dataset. Expand/retrain later.

## Where to place the uploaded dataset

Extract `raw.zip` so your project contains:

```text
data/
└── raw/
    ├── alarm_siren/
    ├── glass_breaking/
    ├── gunshot/
    └── ...other empty class folders are fine
```

## What this script does

1. Reads only the three active starter classes.
2. Ignores `.gitkeep` and unsupported files.
3. Gives every audio file a stable starter Audio ID.
4. Calculates a SHA-256 hash.
5. Excludes exact duplicate recordings **without deleting the raw source**.
6. Excludes originals longer than 10 seconds in starter mode. This prevents the current 360-second gunshot recording from overwhelming the tiny dataset.
7. Makes a deterministic class-wise 70% / 15% / 15% split.
8. Converts included audio to mono, 22,050 Hz, exactly 3 seconds.
9. Short recordings are zero padded.
10. Recordings between 3 and 10 seconds are cropped around their highest-energy region.
11. Writes prepared WAV files into train / validation / test folders.
12. Generates full/included/excluded metadata CSV files and a dataset report.

## Run

From the project root:

```bat
prepare_starter_dataset.bat
```

Or:

```bat
.venv\Scripts\activate
python scripts\prepare_starter_3class_dataset.py --clean
```

## Output

```text
data/starter_3class/
├── train/
├── validation/
├── test/
├── metadata_full.csv
├── metadata_included.csv
└── metadata_excluded.csv

reports/starter_3class/
├── dataset_report.json
└── dataset_report.txt
```

## Important

Do not delete the original `data/raw` recordings. The prepared folder is derived data and can always be rebuilt.

The long 360-second gunshot recording is excluded only in this small starter configuration. Later, when the full dataset is built, it can be segmented into fixed windows **as one original group**, so all derived windows remain in the same train/validation/test split.
