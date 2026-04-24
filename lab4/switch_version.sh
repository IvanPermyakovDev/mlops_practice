#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 lab4-selected|lab4-age-filled|lab4-sex-encoded"
  exit 1
fi

version="$1"

git checkout "$version"
dvc checkout
python lab4/src/inspect_dataset.py lab4/data/titanic.csv
