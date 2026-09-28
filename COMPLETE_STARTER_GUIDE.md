# SonicSentinel AI — Complete Current Starter Project

This bundle contains the current web application plus the current 3-class starter ML dataset/pipeline.

## Included web work

- Flask application
- Home page
- Login / registration
- MySQL integration (default port 3307 via `.env`)
- Profile / logout flow without a separate dashboard page
- Audio upload + preprocessing + quality inspection
- Live Monitoring UI + browser microphone capture + 3-second inspection windows
- Event History / Alerts / Manual Review / Reports route/UI placeholders

## Included starter dataset

Current active training classes only:

1. Alarm or Siren
2. Glass Breaking
3. Gunshot

Raw supplied recordings are included in `data/raw/`.
Prepared data is included in `data/starter_3class/`.

Current usable prepared samples:

- Alarm/Siren: 45
- Glass Breaking: 12
- Gunshot: 44
- Total: 101

Current split:

- Train: 69
- Validation: 16
- Test: 16

Three exact duplicate Glass Breaking files and one 360-second Gunshot outlier are excluded in starter mode. Raw sources are kept.

## Custom model approach

No pretrained Python audio model is used.

Models:

- Random Forest baseline
- SVM baseline
- Custom CNN trained from random initialization on Log-Mel spectrograms

The starter CNN has 3 softmax outputs. When the other seven SRS classes are collected, update the class configuration and retrain all models for the final 10-class system.

## 1. Setup

Use Python 3.11.

```bat
run_setup.bat
```

If setup was already done:

```bat
.venv\Scripts\activate
python app.py
```

## 2. Database

Copy environment configuration:

```bat
copy .env.example .env
```

Default project setting uses MySQL port 3307. Check `.env` before starting.

Initialize database:

```bat
python scripts\init_db.py
```

## 3. Rebuild starter dataset + feature cache

Prepared data is already included, but it can be rebuilt from the included raw files:

```bat
run_dataset_pipeline.bat
```

This performs:

- duplicate filtering
- long starter outlier exclusion
- 70/15/15 split
- mono / 22,050 Hz / 3-second preparation
- split-manifest synchronization
- acoustic feature extraction
- one training-only augmented feature variant per original

## 4. Train starter Random Forest

```bat
python -m src.ml.train_rf --quick
```

## 5. Train starter SVM

```bat
python -m src.ml.train_svm --quick
```

## 6. Install CNN dependency

TensorFlow is large, so keep several GB free disk space.

```bat
python -m pip install -r requirements-ml.txt
```

## 7. Train custom CNN from scratch

Development run:

```bat
python -m src.ml.train_cnn --epochs 10 --batch-size 8
```

Longer run later:

```bat
python -m src.ml.train_cnn --epochs 50 --batch-size 8
```

No pretrained weights are loaded.

## 8. Evaluate and compare

```bat
python scripts\evaluate_models.py --batch-size 8
python scripts\compare_models.py
```

Or use:

```bat
train_starter_models.bat
```

## 9. Test prediction

After at least one model is trained:

```bat
python scripts\predict_audio.py "C:\path\to\audio.wav"
```

## Important

This 3-class model is only a development prototype. The final SRS requires all 10 mandatory sound classes and a much larger balanced original dataset. Do not report starter accuracy as final-system accuracy.
