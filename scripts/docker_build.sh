#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-ece228-pi0-lite-flow:latest}"
INSTALL_LIBERO="${INSTALL_LIBERO:-0}"

docker build \
  --build-arg "INSTALL_LIBERO=${INSTALL_LIBERO}" \
  --tag "${IMAGE}" \
  .
