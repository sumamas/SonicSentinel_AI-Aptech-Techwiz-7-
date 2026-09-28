@echo off
setlocal

echo ==========================================
echo SonicSentinel Model Training
 echo ==========================================

if not exist .venv\Scripts\python.exe (
    echo ERROR: .venv not found.
    pause
    exit /b 1
)

call .venv\Scripts\activate

python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)" >nul 2>&1
if errorlevel 1 (
    echo TensorFlow is not installed.
    echo Run: python -m pip install -r requirements-ml.txt
    pause
    exit /b 1
)

python scripts\train_all_models.py --epochs 50 --batch-size 16 --noise-test
if errorlevel 1 goto :fail

echo.
echo ==========================================
echo Training and evaluation completed.
echo ==========================================
pause
exit /b 0

:fail
echo.
echo MODEL TRAINING FAILED. Read the error above.
pause
exit /b 1
