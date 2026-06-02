#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-ece228-pi0-lite-flow:latest}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
STEPS="${STEPS:-1000}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RUN_NAME="${RUN_NAME:-clip_flow_h16_docker}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p "${ROOT}/runs" "${HF_CACHE}"

docker run --rm \
  --gpus all \
  --ipc=host \
  --user "$(id -u):$(id -g)" \
  --volume "${ROOT}/runs:/workspace/runs" \
  --volume "${HF_CACHE}:/cache/huggingface" \
  --env HF_HOME=/cache/huggingface \
  --env MUJOCO_GL=egl \
  --env PYOPENGL_PLATFORM=egl \
  "${IMAGE}" \
  python scripts/train_flow_custom.py \
    --use-clip \
    --steps "${STEPS}" \
    --max-samples "${MAX_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --lr 1e-4 \
    --save-every 250 \
    --run-name "${RUN_NAME}" \
    "$@"
