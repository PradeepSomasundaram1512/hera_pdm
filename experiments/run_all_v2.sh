#!/bin/bash
# Full v2 compute pipeline (sequential stages; each stage is idempotent).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for DS in femto xjtu; do
  export HERA_DATASET=$DS
  echo "=== $DS tune $(date)"
  $PY run_experiments.py --stage tune
  echo "=== $DS train $(date)"
  $PY -u run_experiments.py --stage train --workers 5
  echo "=== $DS cvplus $(date)"
  $PY -u run_experiments.py --stage cvplus --workers 5
done
unset HERA_DATASET
echo "=== rmax sensitivity $(date)"
$PY -u experiments/rmax_sensitivity.py
echo "=== ALL DONE $(date)"
