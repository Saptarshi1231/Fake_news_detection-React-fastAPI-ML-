#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -r backend/requirements.txt
python backend/prepare_dataset.py
