@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title SonicSentinel AI - Setup

echo ==========================================================
echo   SonicSentinel AI - one click setup
echo ==========================================================
echo.

rem ---- 1. Find Python (3.11 recommended; 3.12 / 3.10 also work) ----
set "PYCMD="
for %%V in (3.11 3.12 3.10) do (
  if not defined PYCMD (
    py -%%V --version >nul 2>&1 && set "PYCMD=py -%%V"
  )
)
if not defined PYCMD (
  python --version >nul 2>&1 && set "PYCMD=python"
)
if not defined PYCMD (
  echo [ERROR] Python was not found.
  echo         Install Python 3.11 from https://www.python.org/downloads/
  echo         and tick "Add python.exe to PATH" during installation.
  pause
  exit /b 1
)
echo [1/6] Using Python: %PYCMD%
%PYCMD% --version

rem ---- 2. Virtual environment ----
if not exist ".venv\Scripts\python.exe" (
  echo [2/6] Creating virtual environment .venv ...
  %PYCMD% -m venv .venv
  if errorlevel 1 goto :failed
) else (
  echo [2/6] Virtual environment already exists.
)
set "VPY=.venv\Scripts\python.exe"

rem ---- 3. Packages ----
echo [3/6] Installing packages (first time can take 5-15 minutes, TensorFlow is large) ...
"%VPY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :failed
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

rem ---- 4. Configuration ----
if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo [4/6] Created .env from .env.example
) else (
  echo [4/6] .env already exists - keeping your settings.
)
if not exist "uploads" mkdir uploads
if not exist "instance" mkdir instance

rem ---- 5. Database ----
echo [5/6] Preparing database (MySQL on port 3307 if XAMPP is running, otherwise SQLite) ...
"%VPY%" scripts\init_db.py
if errorlevel 1 goto :failed

rem ---- 6. Self test ----
echo [6/6] Checking both models ...
"%VPY%" scripts\check_setup.py

echo.
echo ==========================================================
echo   Setup complete.
echo   Start the app any time with:  start_project.bat
echo   Admin login   : admin@sonicsentinel.local  /  Admin@12345
echo   Reviewer login: reviewer@sonicsentinel.local  /  Review@12345
echo ==========================================================
echo.
choice /C YN /M "Start SonicSentinel now"
if errorlevel 2 exit /b 0
call start_project.bat
exit /b 0

:failed
echo.
echo [ERROR] Setup failed. Read the message above.
echo  - "No matching distribution": use Python 3.11 (64-bit).
echo  - Network errors: check your internet connection and run again.
pause
exit /b 1
