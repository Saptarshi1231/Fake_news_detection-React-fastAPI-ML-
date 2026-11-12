@echo off
cd /d "%~dp0\.."
if not exist backend\models\fake_news_model.joblib (
  echo Model missing. Run scripts\setup_and_train.ps1 first.
  pause
  exit /b 1
)
if not exist frontend\node_modules (
  echo Frontend dependencies missing. Run scripts\setup_and_train.ps1 first.
  pause
  exit /b 1
)
start "Verity FastAPI" cmd /k "call .venv\Scripts\activate.bat && uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000"
start "Verity React" cmd /k "cd /d frontend && npm run dev -- --host 127.0.0.1"
echo FastAPI: http://127.0.0.1:8000/docs
echo React:   http://localhost:5173
