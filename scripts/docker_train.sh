#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-ece228-pi0-lite-flow:latest}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
STEPS="${STEPS:-1000}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RUN_NAME="${RUN_NAME:-clip_flow_h16_docker}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
GPU_ID="${GPU_ID:-all}"
GPU_REQUEST="${GPU_ID}"
if [[ "${GPU_ID}" != "all" ]]; then
  GPU_REQUEST="device=${GPU_ID}"
fi

mkdir -p "${ROOT}/runs" "${HF_CACHE}"

check_writable_dir() {
  local dir="$1"
  local probe="${dir}/.ece228_write_test"
  if [[ -d "${dir}" ]] && ! touch "${probe}" 2>/dev/null; then
    echo "ERROR: ${dir} is not writable by user $(id -u):$(id -g)." >&2
    echo "Fix on the host with:" >&2
    echo "  sudo chown -R $(id -u):$(id -g) \"${HF_CACHE}\"" >&2
    exit 1
  fi
  rm -f "${probe}" 2>/dev/null || true
}

mkdir -p "${HF_CACHE}/hub/.locks" 2>/dev/null || true
check_writable_dir "${HF_CACHE}"
check_writable_dir "${HF_CACHE}/hub"
check_writable_dir "${HF_CACHE}/hub/.locks"
if [[ -d "${HF_CACHE}/hub/.locks/datasets--HuggingFaceVLA--libero" ]]; then
  check_writable_dir "${HF_CACHE}/hub/.locks/datasets--HuggingFaceVLA--libero"
fi

docker run --rm \
  --gpus "${GPU_REQUEST}" \
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
    --save-every "${SAVE_EVERY}" \
    --run-name "${RUN_NAME}" \
    "$@"
