#!/usr/bin/env bash
# Created: 2026-09-27
# Last updated: 2026-09-27
#
# Gradient-norm probe that sets the global_clipnorm threshold of the cosine-schedule runs,
# decided before any of those runs started:
#
#   d7 u280 reset_after=False, 50M (FORMAL 15M + TRAINONLY seed43 5M + seed44 30M), seed 0,
#   batch 10,000, cosine LR 3e-3 -> 3e-5 laid out over 80 epochs (400,000 updates), NO
#   clipping, determinism on, TF32 on, 1.6 GB GPU cap; stopped after 500 updates.
#   The global gradient norm (before clipping) of every update is recorded.
#
#   threshold = 5 x median of the norms of updates 100-500 (1-indexed, inclusive),
#   rounded to 2 significant figures (pick_threshold.py).
#
#     bash run_probe.sh      (from this folder, on the GPU host)
set -uo pipefail
REPO="$HOME/FNAL-AI4QC-GRU"
DIR="$(cd "$(dirname "$0")" && pwd)"
P7="$HOME/pools_d7_p004_19M/data_d7_p0.004_r7"
S=$(date -u +%Y%m%dT%H%M%SZ); OUT="$DIR/probe_$S"; mkdir -p "$OUT"
echo "repo_head $(git -C "$REPO" rev-parse HEAD)" > "$OUT/code_commit.txt"
podman run --rm --name gclip_probe --security-opt=label=disable --device nvidia.com/gpu=all \
  -v "$HOME:$HOME" -e HOME="$HOME" -e TF_DETERMINISTIC_OPS=1 -e TF_CUDNN_DETERMINISTIC=1 \
  -w "$REPO/src" qdec:tf215 \
  python3 train_student.py --student gru --inputs evts --hidden --units 280 \
    --d 7 --p 0.004 --rounds 7 --pool "${P7}_FORMAL.npz" --n-train 15000000 \
    --extra-train-pool "${P7}_TRAINONLY_seed43.npz" "${P7}_TRAINONLY_seed44.npz" \
    --extra-train-n 5000000 30000000 \
    --val-start 15000000 --val-n 200000 --eval-start 15200000 --n-test 20000 \
    --alpha 1.0 --temperature 1.0 --batch-size 10000 \
    --lr 0.003 --lr-schedule cosine --lr-end 3e-5 --epochs 80 --no-early-stopping \
    --log-grad-norm --stop-after-steps 500 \
    --seed 0 --require-determinism --gpu-mem-mib 1600 \
    --ckpt-dir "$OUT/ckpt" --out-dir "$OUT/runs" --tag probe_u280_50M_s0 \
    > "$OUT/run.log" 2>&1
echo "exit $?" >> "$OUT/run.log"
podman run --rm --security-opt=label=disable -v "$HOME:$HOME" -e CUDA_VISIBLE_DEVICES= \
  -w "$DIR" qdec:tf215 python3 pick_threshold.py "$OUT/runs/probe_u280_50M_s0.grad_norms.npy" \
  --out "$OUT/threshold.json" 2>&1 | grep -v -e NVIDIA -e nvidia -e '^=' -e TensorFlow -e Copyright -e 'By pulling' -e '^$' | tee "$OUT/threshold.log"
