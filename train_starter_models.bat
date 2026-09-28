@echo off
setlocal

echo ================================================
echo SonicSentinel Starter 3-Class Model Training
echo ================================================

if not exist .venv\Scripts\python.exe (
    echo ERROR: .venv not found.
    pause
    exit /b 1
)
call .venv\Scripts\activate

if not exist data\features\train.npz (
    echo Feature cache not found. Running dataset pipeline first...
    call run_dataset_pipeline.bat
    if errorlevel 1 exit /b 1
)

python -m src.ml.train_rf --quick
if errorlevel 1 goto :fail

python -m src.ml.train_svm --quick
if errorlevel 1 goto :fail

echo.
echo Random Forest and SVM training complete.
echo.
python -c "import tensorflow" >nul 2>&1
if errorlevel 1 (
    echo TensorFlow is not installed, so CNN was skipped.
    echo Install it with:
    echo   python -m pip install -r requirements-ml.txt
    echo Then run:
    echo   python -m src.ml.train_cnn --epochs 10 --batch-size 8
    echo.
    pause
    exit /b 0
)

python -m src.ml.train_cnn --epochs 10 --batch-size 8
if errorlevel 1 goto :fail

python scripts\evaluate_models.py --batch-size 8
if errorlevel 1 goto :fail

python scripts\compare_models.py
if errorlevel 1 goto :fail

echo.
echo All starter models trained and evaluated.
pause
exit /b 0

:fail
echo.
echo TRAINING FAILED. Read the error above.
pause
exit /b 1
