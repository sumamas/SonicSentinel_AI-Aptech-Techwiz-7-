@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv
  if errorlevel 1 goto fail
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
echo Start MySQL in XAMPP before continuing. Default database port is 3307.
pause
".venv\Scripts\python.exe" scripts\init_db.py
if errorlevel 1 goto fail
".venv\Scripts\python.exe" scripts\check_project_model.py --database
if errorlevel 1 goto fail
echo Setup complete. Open start_project.bat next.
pause
exit /b 0
:fail
echo Setup failed. Read the error above; do not continue until it is resolved.
pause
exit /b 1
