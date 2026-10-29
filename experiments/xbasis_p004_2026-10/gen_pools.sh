#!/usr/bin/env bash
# Created: 2026-10-29
# Last modified: 2026-10-29
# X-basis FORMAL pools at p = 0.004 (surface_code:rotated_memory_x), 19M shots in the
# configs/partitions.yaml layout, detection events and flips only. Seeds 64 / 65 / 66 for
# d5 / d7 / d9. CPU only. Skips a pool that already exists.
#   bash gen_pools.sh 5 7 9
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../../src" && pwd)"
OUTDIR=${POOLS:-/tmp/kchey_xbasis/pools}
LOG="$HERE/gen_pools.log"
mkdir -p "$OUTDIR"
for d in "$@"; do
  case "$d" in 5) seed=64 ;; 7) seed=65 ;; 9) seed=66 ;; *) echo "distance 5, 7 or 9"; exit 1 ;; esac
  f="$OUTDIR/data_d${d}_p0.004_r${d}_X_FORMAL_seed${seed}.npz"
  [ -s "${f%.npz}.fingerprint.json" ] && { echo "[pools] $f exists" >> "$LOG"; continue; }
  rm -f "$f"     # a partial file from an interrupted run
  echo "[pools $(date -u +%FT%TZ)] START d=$d basis=x seed=$seed -> $f (free $(df -BG --output=avail "$OUTDIR" | tail -1 | tr -d ' '))" >> "$LOG"
  podman run --rm --security-opt=label=disable -v "$HOME:$HOME" -v /tmp/kchey_xbasis:/tmp/kchey_xbasis \
    -e CUDA_VISIBLE_DEVICES= -w "$SRC" qdec:tf215 \
    python3 gen_train_extension.py --d $d --p 0.004 --rounds $d --n 19000000 --gen-seed $seed \
    --out-dir "$OUTDIR" --formal --basis x --evts-only 2>&1 | grep "^\[ext\]" >> "$LOG"
  echo "[pools $(date -u +%FT%TZ)] END   d=$d seed=$seed exit ${PIPESTATUS[0]}" >> "$LOG"
done
