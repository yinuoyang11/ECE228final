#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-ece228-pi0-lite-flow:latest}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
TORCH_CACHE="${TORCH_CACHE:-${ROOT}/.torch_cache}"
STEPS="${STEPS:-1000}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
RUN_NAME="${RUN_NAME:-qwenvl_task20}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
GPU_ID="${GPU_ID:-all}"
GPU_REQUEST="${GPU_ID}"
if [[ "${GPU_ID}" != "all" ]]; then
  GPU_REQUEST="device=${GPU_ID}"
fi

mkdir -p "${ROOT}/runs" "${HF_CACHE}" "${HF_CACHE}/hub/.locks" "${TORCH_CACHE}" "${TORCH_CACHE}/kernels"

check_writable_dir() {
  local dir="$1"
  local probe="${dir}/.ece228_write_test"
  if [[ -d "${dir}" ]] && ! touch "${probe}" 2>/dev/null; then
    echo "ERROR: ${dir} is not writable by user $(id -u):$(id -g)." >&2
    echo "Fix by using a writable cache, for example:" >&2
    echo "  HF_CACHE=\"${ROOT}/.hf_cache\" bash scripts/docker_train_qwenvl.sh" >&2
    exit 1
  fi
  rm -f "${probe}" 2>/dev/null || true
}

check_writable_dir "${HF_CACHE}"
check_writable_dir "${HF_CACHE}/hub"
check_writable_dir "${HF_CACHE}/hub/.locks"
check_writable_dir "${TORCH_CACHE}"
check_writable_dir "${TORCH_CACHE}/kernels"

docker run --rm \
  --gpus "${GPU_REQUEST}" \
  --ipc=host \
  --user "$(id -u):$(id -g)" \
  --volume "${ROOT}/runs:/workspace/runs" \
  --volume "${HF_CACHE}:/cache/huggingface" \
  --volume "${TORCH_CACHE}:/cache/torch" \
  --env HOME=/workspace \
  --env HF_HOME=/cache/huggingface \
  --env TORCH_HOME=/cache/torch \
  --env XDG_CACHE_HOME=/cache \
  --env PYTORCH_KERNEL_CACHE_PATH=/cache/torch/kernels \
  --env MUJOCO_GL=egl \
  --env PYOPENGL_PLATFORM=egl \
  "${IMAGE}" \
  python scripts/train_qwenvl_flow_custom.py \
    --steps "${STEPS}" \
    --max-samples "${MAX_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
    --num-workers "${NUM_WORKERS}" \
    --save-every "${SAVE_EVERY}" \
    --run-name "${RUN_NAME}" \
    "$@"
