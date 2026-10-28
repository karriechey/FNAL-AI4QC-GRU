#!/usr/bin/env bash
# Created: 2026-10-28 | Last updated: 2026-10-28
#   bash pod.sh <container name> python3 <script> [args]     (GPU container, from this folder)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
name=$1; shift
mkdir -p /tmp/kchey_qat
podman run --rm --name "$name" --security-opt=label=disable --device nvidia.com/gpu=all \
  -v "$HOME:$HOME" -v /tmp/kchey_qat:/tmp/kchey_qat -e HOME="$HOME" \
  -e TF_DETERMINISTIC_OPS=1 -e TF_CUDNN_DETERMINISTIC=1 -e MALLOC_ARENA_MAX=2 -e TF_CPP_MIN_LOG_LEVEL=2 \
  -w "$HERE" qdec:tf215 "$@"
