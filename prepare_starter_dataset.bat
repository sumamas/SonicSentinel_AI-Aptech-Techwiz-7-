@echo off
setlocal

echo ================================================
echo SonicSentinel - Prepare Starter 3-Class Dataset
echo ================================================
echo.

if not exist .venv\Scripts\python.exe (
    echo ERROR: .venv not found.
    echo Create/activate your Python 3.11 project environment first.
    pause
    exit /b 1
)

call .venv\Scripts\activate

python scripts\prepare_starter_3class_dataset.py --clean
if errorlevel 1 (
    echo.
    echo DATASET PREPARATION FAILED.
    pause
    exit /b 1
)

echo.
echo Dataset preparation completed successfully.
echo.
pause
