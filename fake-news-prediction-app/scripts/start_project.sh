#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -f backend/models/fake_news_model.joblib ]]; then
  echo "Model missing. Run ./scripts/setup_and_train.sh first." >&2
  exit 1
fi
if [[ ! -d frontend/node_modules ]]; then
  echo "Frontend dependencies missing. Run ./scripts/setup_and_train.sh first." >&2
  exit 1
fi
source .venv/bin/activate
uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 &
API_PID=$!
cleanup() { kill "$API_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
cd frontend
npm run dev -- --host 127.0.0.1
