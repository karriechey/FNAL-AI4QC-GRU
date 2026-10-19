#!/usr/bin/env bash
# Created: 2026-10-14
# Last modified: 2026-10-14
#
# p = 0.003 FORMAL pools for d5 and d9 (d7 has one: seed 52), for the per-distance winners
# trained on fresh shots (experiments/ceiling_probe_d7_2026-10-05/run_arm10.sh). Fresh-shot
# runs need no training-only pools; the FORMAL pool holds the validation block, the evaluation
# block and the sealed block (configs/partitions.yaml layout). Same generator and image as
# experiments/queue12_pools_2026-09-30/generate.sh (CPU only). New generation seeds:
#   d5 p=0.003: 60      d9 p=0.003: 61      (42-59, 7001, 7002 are taken)
# Sizes: d5 about 6 GB, d9 about 27 GB (full format: measurements, det_evts, flips).
#
#     bash generate.sh 5 [9]        (on grace1, from this folder; one argument per distance)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../../src" && pwd)"
LOG="$HERE/generate.log"
for d in "$@"; do
  case "$d" in 5) seed=60 ;; 9) seed=61 ;; *) echo "distance 5 or 9"; exit 1 ;; esac
  out="$HOME/pools_d${d}_p003_19M"
  echo "[pools $(date -u +%FT%TZ)] START d=$d p=0.003 n=19000000 seed=$seed --formal -> $out (free $(df -BG --output=avail "$HOME" | tail -1 | tr -d ' '))" | tee -a "$LOG"
  podman run --rm --security-opt=label=disable -v "$HOME:$HOME" -e CUDA_VISIBLE_DEVICES= -w "$SRC" qdec:tf215 \
    python3 gen_train_extension.py --d $d --p 0.003 --rounds $d --n 19000000 --gen-seed $seed --out-dir "$out" --formal \
    2>&1 | grep "^\[ext\]" >> "$LOG"
  echo "[pools $(date -u +%FT%TZ)] END   d=$d p=0.003 seed=$seed exit ${PIPESTATUS[0]}" | tee -a "$LOG"
done
