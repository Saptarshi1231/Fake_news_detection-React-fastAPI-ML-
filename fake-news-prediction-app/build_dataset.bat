@echo off
cd /d "%~dp0"
python -m pip install -r backend\requirements.txt
if errorlevel 1 exit /b 1
python backend\prepare_dataset.py
if errorlevel 1 exit /b 1
echo.
echo Dataset created at backend\data\train.csv
pause
