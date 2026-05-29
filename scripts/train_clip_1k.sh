#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

"${PYTHON}" scripts/train_flow_custom.py \
  --use-clip \
  --steps 1000 \
  --max-samples 5000 \
  --batch-size 8 \
  --lr 1e-4 \
  --save-every 250 \
  --run-name clip_flow_h16_1k

