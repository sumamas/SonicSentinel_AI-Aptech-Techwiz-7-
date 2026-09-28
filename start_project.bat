@echo off
setlocal
cd /d "%~dp0"
title SonicSentinel AI
if not exist ".venv\Scripts\python.exe" (
  echo Setup has not been run yet. Starting run_setup.bat ...
  call run_setup.bat
  exit /b
)
echo ==========================================================
echo   SonicSentinel AI is starting at http://127.0.0.1:5000
echo   (XAMPP MySQL is optional - SQLite is used if it is off)
echo   Press Ctrl+C in this window to stop the server.
echo ==========================================================
start "" cmd /c "timeout /t 6 >nul & start http://127.0.0.1:5000"
set TF_CPP_MIN_LOG_LEVEL=2
".venv\Scripts\python.exe" -m flask --app app run --host 127.0.0.1 --port 5000
pause
