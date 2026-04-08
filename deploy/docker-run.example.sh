#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/srv/survival_wrappers"
IMAGE_NAME="sawrap:latest"
CONTAINER_NAME="sawrap"

cd "${APP_DIR}"

git pull --ff-only

docker build -t "${IMAGE_NAME}" .

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker rm -f "${CONTAINER_NAME}"
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  --cpus="0.40" \
  --memory="800m" \
  --pids-limit=128 \
  -e SAWRAP_SKIP_MISSING_RECALC=1 \
  -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e NUMEXPR_NUM_THREADS=1 \
  -p 127.0.0.1:8001:8000 \
  "${IMAGE_NAME}"

docker ps --filter "name=${CONTAINER_NAME}"
