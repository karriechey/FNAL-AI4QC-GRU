#!/usr/bin/env bash
# Created: 2026-09-30
# New pools for the twelve-run queue, generated on CPU (qdec:tf215 image, no GPU) with
# src/gen_train_extension.py from the compact-pools branch. Order: d7 +100M (run 1),
# d9 +85M (runs 2-5), d5 +80M (run 8), p=0.003 and p=0.005 FORMAL + 85M (runs 11-12).
# Training-only pools are compact (bit-packed detection events); FORMAL pools are full.
# Generation seeds are new (used before: 42 FORMAL, 43, 44, 45).
set -uo pipefail
W="$HOME/FNAL-AI4QC-GRU-compact"; G="$W/experiments/queue12_pools_2026-09-30"
gen () {  # gen <d> <p> <n> <seed> <out dir> [--compact|--formal]
  local t0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[pools $t0] START d=$1 p=$2 n=$3 seed=$4 $6 -> $5" | tee -a "$G/generate.log"
  podman run --rm --security-opt=label=disable -v "$HOME:$HOME" -e CUDA_VISIBLE_DEVICES= -w "$W/src" qdec:tf215 \
    python3 gen_train_extension.py --d $1 --p $2 --rounds $1 --n $3 --gen-seed $4 --out-dir "$5" $6 \
    2>&1 | grep "^\[ext\]" >> "$G/generate.log"
  echo "[pools $(date -u +%Y-%m-%dT%H:%M:%SZ)] END   d=$1 p=$2 seed=$4 exit ${PIPESTATUS[0]}" | tee -a "$G/generate.log"
}
gen 7 0.004 50000000 46 "$HOME/pools_d7_p004_19M" --compact
gen 7 0.004 50000000 47 "$HOME/pools_d7_p004_19M" --compact
gen 9 0.004 35000000 48 "$HOME/pools_d9_p004_19M" --compact
gen 9 0.004 50000000 49 "$HOME/pools_d9_p004_19M" --compact
gen 5 0.004 30000000 50 "$HOME/pools_d5_p004_19M" --compact
gen 5 0.004 50000000 51 "$HOME/pools_d5_p004_19M" --compact
gen 7 0.003 19000000 52 "$HOME/pools_d7_p003_19M" --formal
gen 7 0.003 35000000 53 "$HOME/pools_d7_p003_19M" --compact
gen 7 0.003 50000000 54 "$HOME/pools_d7_p003_19M" --compact
gen 7 0.005 19000000 55 "$HOME/pools_d7_p005_19M" --formal
gen 7 0.005 35000000 56 "$HOME/pools_d7_p005_19M" --compact
gen 7 0.005 50000000 57 "$HOME/pools_d7_p005_19M" --compact
echo "[pools $(date -u +%Y-%m-%dT%H:%M:%SZ)] ALL DONE" | tee -a "$G/generate.log"
