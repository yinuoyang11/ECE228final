#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# docker_rollout.sh
#
# Build the Docker image and run simulation rollout for BC Regression.
# Must be run on a Linux machine with Docker + NVIDIA Container Toolkit.
#
# Usage:
#   bash scripts/docker_rollout.sh                    # default: tasks 20+21+22, 10 eps each
#   TASK_INDICES="20" EPISODES=5 bash scripts/docker_rollout.sh
#
# Environment variables (all optional):
#   TASK_INDICES   Space-separated dataset task indices  (default: "20 21 22")
#   EPISODES       Episodes per task                     (default: 10)
#   SUITE          LIBERO benchmark suite                (default: libero_object)
#   CHECKPOINT     Path to checkpoint inside container   (default: below)
#   DEVICE         cuda or cpu                           (default: cuda)
#   IMAGE          Docker image name                     (default: ece228-regression)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

IMAGE="${IMAGE:-ece228-regression}"
TASK_INDICES="${TASK_INDICES:-20 21 22}"
EPISODES="${EPISODES:-10}"
SUITE="${SUITE:-libero_object}"
CHECKPOINT="${CHECKPOINT:-results/task202122_regression/checkpoint_final.pt}"
DEVICE="${DEVICE:-cuda}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "======================================================="
echo "  BC Regression — LIBERO Simulation Rollout"
echo "======================================================="
echo "  Image:        $IMAGE"
echo "  Tasks:        $TASK_INDICES"
echo "  Episodes:     $EPISODES per task"
echo "  Suite:        $SUITE"
echo "  Checkpoint:   $CHECKPOINT"
echo "  Device:       $DEVICE"
echo "  Results dir:  $REPO_ROOT/results"
echo "======================================================="

# ── 1. Build image ────────────────────────────────────────────────────────────
echo ""
echo "[1/2] Building Docker image: $IMAGE ..."
docker build -t "$IMAGE" "$REPO_ROOT"

# ── 2. Run rollout ────────────────────────────────────────────────────────────
echo ""
echo "[2/2] Running simulation rollout ..."
docker run --gpus all --rm \
    -v "$REPO_ROOT/results":/workspace/results:rw \
    "$IMAGE" \
    python scripts/sim_rollout_regression.py \
        "$CHECKPOINT" \
        --dataset-task-indices $TASK_INDICES \
        --episodes-per-task "$EPISODES" \
        --suite "$SUITE" \
        --video-dir results/sim_rollout \
        --device "$DEVICE"

echo ""
echo "Done. Results saved to: $REPO_ROOT/results/sim_rollout/"
echo "  rollout_metrics.json  ← success_rate is here"
echo "  *.mp4                 ← one video per episode"
