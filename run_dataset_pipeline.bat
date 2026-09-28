@echo off
setlocal

echo =====================================================
echo SonicSentinel Starter 3-Class Dataset + Features
echo =====================================================
echo.

if not exist .venv\Scripts\python.exe (
    echo ERROR: .venv not found. Run run_setup.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate

python scripts\prepare_starter_3class_dataset.py --clean
if errorlevel 1 goto :fail

python scripts\sync_starter_splits.py
if errorlevel 1 goto :fail

python scripts\build_feature_cache.py --augment-train 1
if errorlevel 1 goto :fail

echo.
echo =====================================================
echo Starter dataset pipeline completed successfully.
echo =====================================================
echo.
echo Next commands:
echo   python -m src.ml.train_rf --quick
echo   python -m src.ml.train_svm --quick
echo   python -m pip install -r requirements-ml.txt
echo   python -m src.ml.train_cnn --epochs 10 --batch-size 8
echo.
pause
exit /b 0

:fail
echo.
echo STARTER DATASET PIPELINE FAILED. Read the error above.
pause
exit /b 1
