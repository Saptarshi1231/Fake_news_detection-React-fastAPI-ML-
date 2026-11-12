#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
python backend/prepare_dataset.py
python backend/train_model.py
( cd frontend && npm install )
echo "Setup complete. Start the app with ./scripts/start_project.sh"
