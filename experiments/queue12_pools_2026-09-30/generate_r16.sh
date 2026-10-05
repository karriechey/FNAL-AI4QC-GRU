#!/usr/bin/env bash
# Created: 2026-10-01. Extra d9 training-only pools for run 16 (d9 200M): seeds 58, 59, 50M each, compact.
set -uo pipefail
W="$HOME/FNAL-AI4QC-GRU-compact"; G="$W/experiments/queue12_pools_2026-09-30"
for s in 58 59; do
  echo "[pools $(date -u +%Y-%m-%dT%H:%M:%SZ)] START d=9 p=0.004 n=50000000 seed=$s --compact" | tee -a "$G/generate.log"
  podman run --rm --security-opt=label=disable -v "$HOME:$HOME" -e CUDA_VISIBLE_DEVICES= -w "$W/src" qdec:tf215 \
    python3 gen_train_extension.py --d 9 --p 0.004 --rounds 9 --n 50000000 --gen-seed $s --out-dir "$HOME/pools_d9_p004_19M" --compact 2>&1 | grep "^\[ext\]" >> "$G/generate.log"
  echo "[pools $(date -u +%Y-%m-%dT%H:%M:%SZ)] END   d=9 seed=$s exit ${PIPESTATUS[0]}" | tee -a "$G/generate.log"
done
