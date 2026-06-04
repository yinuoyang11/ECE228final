#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

"${PYTHON}" scripts/train_flow_custom.py \
  --use-clip \
  --steps 100 \
  --max-samples 512 \
  --batch-size 4 \
  --lr 1e-4 \
  --save-every 50 \
  --run-name clip_flow_h16_debug

