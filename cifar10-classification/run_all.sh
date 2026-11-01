#!/usr/bin/env bash
# Train and evaluate both models, then compare them. Extra arguments are passed to train.py,
# e.g. ./run_all.sh --epochs 10
set -euo pipefail
cd "$(dirname "$0")"

for model in baseline residual; do
  python train.py --model "$model" "$@"
  python evaluate.py --model "$model"
done
python compare.py
