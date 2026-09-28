# SonicSentinel AI — Dataset Preparation + Custom Model Training

This pipeline trains SonicSentinel without any pretrained audio model or transfer-learning weights.

## Models

1. Random Forest — classical baseline using handcrafted acoustic features.
2. SVM — classical baseline using the same acoustic feature vector.
3. Custom CNN — designed in this repository and trained from random initialization on SonicSentinel Log-Mel spectrograms.

The Google Teachable Machine model is a separate SRS requirement and is not used to train or influence the custom Python models in this pipeline.

## Fixed audio standard

- Sample rate: 22,050 Hz
- Mono
- Model window: 3.0 seconds
- CNN input: 128 x 130 normalized Log-Mel spectrogram, 1 channel
- Random seed: 42

## Mandatory classes

Place original recordings in these exact folders:

```text
data/raw/
├── machinery_fault/
├── glass_breaking/
├── alarm_siren/
├── vehicle_horn/
├── animal_sound/
├── gunshot/
├── panic_scream/
├── aggression/
├── person_asking_for_help/
└── background_noise/
```

Recommended development target: 20–30 unique originals per class so the pipeline can be tested.
Final SRS target: approximately 300 unique original clips per class, 3,000 total.

Do not put the same recording, a trimmed copy, or a re-encoded copy into multiple classes/splits. Use legally/ethically sourced audio only.

## Suggested naming

```text
MF_0001.wav
GB_0001.wav
AS_0001.wav
VH_0001.wav
AN_0001.wav
GS_0001.wav
PS_0001.wav
AG_0001.wav
PH_0001.wav
BN_0001.wav
```

Names are not used as labels; the folder name supplies the label.

## 0. Activate the project environment

```bat
.venv\Scripts\activate
python --version
```

Use Python 3.11.x.

The existing Flask/audio dependencies are installed with:

```bat
python -m pip install -r requirements.txt
```

Before CNN training, install TensorFlow:

```bat
python -m pip install -r requirements-ml.txt
```

TensorFlow is large. Make sure the drive has several GB of free disk space before installing.

## 1. Add raw audio

Copy your labelled recordings into `data/raw/<class-folder>/`.

For the first development run, use at least 20 unique usable recordings in every class. The split script requires at least 7 unique originals per class, but 20+ is strongly preferred for a meaningful smoke test.

## 2. Create metadata

```bat
python scripts\create_metadata.py
```

Outputs:

- `data/metadata.csv`
- `reports/metadata_failures.csv`

Metadata includes Audio ID, class, path, duration, sample rate, channels, file size, SHA-256 hash, duplicate link, and fields for source/environment/device/distance.

After generation, you can manually fill `source`, `environment`, `device`, and `distance` in `data/metadata.csv`. Re-running the metadata script preserves those manual fields for unchanged paths.

## 3. Validate the dataset

Development check:

```bat
python scripts\validate_dataset.py --minimum-per-class 20
```

Final SRS check:

```bat
python scripts\validate_dataset.py --minimum-per-class 300
```

Outputs:

- `reports/dataset_validation.csv`
- `reports/dataset_validation_summary.json`

The validator checks decoding, duration, signal level, class counts and exact duplicate hashes.

## 4. Create the 70/15/15 split

```bat
python scripts\split_dataset.py
```

Outputs:

```text
data/splits/train.csv
data/splits/validation.csv
data/splits/test.csv
```

Exact duplicate hashes are excluded from model splits. Splitting happens on original recordings before augmentation. Augmented variants are created only from training recordings.

## 5. Build RF/SVM feature cache

```bat
python scripts\build_feature_cache.py --augment-train 1
```

`--augment-train 1` creates one in-memory augmented version per training original. Validation/test data are never augmented.

Acoustic features include:

- 40 MFCCs (mean/std)
- Log-Mel summary features
- Chroma
- Zero-crossing rate
- RMS energy
- Spectral centroid
- Spectral bandwidth
- Spectral roll-off
- Onset strength
- Tempo

Outputs:

```text
data/features/train.npz
data/features/validation.npz
data/features/test.npz
data/features/feature_cache_summary.json
```

## 6. Train Random Forest

Development/quick tuning:

```bat
python -m src.ml.train_rf --quick
```

Full tuning:

```bat
python -m src.ml.train_rf
```

Outputs include:

```text
models/random_forest.joblib
reports/model_evaluation/random_forest_validation_metrics.json
reports/model_evaluation/random_forest_validation_class_metrics.csv
reports/model_evaluation/random_forest_validation_confusion_matrix.png
```

## 7. Train SVM

Development/quick tuning:

```bat
python -m src.ml.train_svm --quick
```

Full tuning:

```bat
python -m src.ml.train_svm
```

Outputs include:

```text
models/svm.joblib
reports/model_evaluation/svm_validation_metrics.json
reports/model_evaluation/svm_validation_class_metrics.csv
reports/model_evaluation/svm_validation_confusion_matrix.png
```

## 8. Train the custom CNN from scratch

Install TensorFlow first if necessary:

```bat
python -m pip install -r requirements-ml.txt
```

Development run:

```bat
python -m src.ml.train_cnn --epochs 10 --batch-size 16
```

Full run:

```bat
python -m src.ml.train_cnn --epochs 50 --batch-size 16
```

Architecture:

```text
128x130x1 Log-Mel input
  ↓
Conv2D 32 + BatchNorm + ReLU + MaxPool + Dropout
  ↓
Conv2D 64 + BatchNorm + ReLU + MaxPool + Dropout
  ↓
Conv2D 128 + BatchNorm + ReLU + MaxPool + Dropout
  ↓
Conv2D 256 + BatchNorm + ReLU + MaxPool + Dropout
  ↓
Global Average Pooling
  ↓
Dense 128 + ReLU + Dropout
  ↓
Dense 10 + Softmax
```

No pretrained model, transfer-learning architecture, or pretrained weights are loaded.

Outputs:

```text
models/custom_cnn.keras
models/custom_cnn_last.keras
models/custom_cnn_manifest.json
reports/model_evaluation/custom_cnn_training_history.csv
reports/model_evaluation/custom_cnn_training_history.png
reports/model_evaluation/custom_cnn_validation_metrics.json
reports/model_evaluation/custom_cnn_validation_confusion_matrix.png
```

## 9. Evaluate all models on the untouched test set

Clean test evaluation:

```bat
python scripts\evaluate_models.py
```

Clean + controlled noise robustness at 20, 10 and 5 dB SNR:

```bat
python scripts\evaluate_models.py --noise
```

Outputs include accuracy, macro precision, macro recall, macro F1, class-wise metrics, critical-event recall and confusion matrices. Noise testing also creates:

```text
reports/model_evaluation/noise_robustness.csv
reports/model_evaluation/noise_robustness_summary.csv
```

## 10. Compare and select the Python model

```bat
python scripts\compare_models.py
```

Outputs:

```text
reports/model_evaluation/model_comparison.csv
models/selected_model.json
```

The automated comparison uses multiple clean-test metrics and includes noise robustness when that report exists. Always inspect the confusion matrices and class-wise results before accepting the deployment choice.

## 11. Test one audio file

Use the selected model:

```bat
python scripts\predict_audio.py "C:\path\to\test.wav"
```

Or choose a specific model:

```bat
python scripts\predict_audio.py "C:\path\to\test.wav" --model random_forest
python scripts\predict_audio.py "C:\path\to\test.wav" --model svm
python scripts\predict_audio.py "C:\path\to\test.wav" --model custom_cnn
```

Output contains the predicted class, top-3 predictions and confidence for every supported class.

## One-click dataset preparation

After adding audio files:

```bat
run_dataset_pipeline.bat
```

It runs metadata -> validation -> split -> feature cache.

## One-click training

After TensorFlow is installed and the dataset pipeline succeeds:

```bat
train_models.bat
```

It trains RF, SVM and Custom CNN, evaluates them on the test set, runs noise robustness testing and creates the comparison/selection manifest.

For a quick development run instead of the batch file:

```bat
python scripts\train_all_models.py --quick --epochs 10 --batch-size 16
```

## Important data-leakage rules

- Split original recordings before augmentation.
- Never augment validation or test data.
- All segments/variants derived from one original remain associated with that original's split.
- Never tune hyperparameters using the final test set.
- Test data are used only after model development/tuning is complete.
- Exact duplicates must not appear across splits.

## Expected project state after this pipeline

```text
Raw labelled audio
    ↓
Metadata + SHA-256 duplicate checks
    ↓
70/15/15 stratified split
    ↓
Training-only augmentation
    ↓
Handcrafted features ───────→ Random Forest
          │                  → SVM
          │
          └→ Log-Mel spectrogram → Custom CNN from scratch
                                      ↓
                             Validation / tuning
                                      ↓
                             Untouched test set
                                      ↓
                       Metrics + confusion matrices
                                      ↓
                         Noise robustness testing
                                      ↓
                             Model comparison
                                      ↓
                            selected_model.json
                                      ↓
                              Flask integration
```
