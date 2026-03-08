#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${ROOT_DIR}/.venv-lab2"
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r lab2/requirements.txt

python lab2/data_collection.py
python lab2/data_preprocessing.py
python lab2/model_training.py
python lab2/model_evaluation.py
