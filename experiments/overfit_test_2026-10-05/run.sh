#!/usr/bin/env bash
# Created: 2026-10-05
# Last modified: 2026-10-05
#
# Overfit test (requested 2026-10-05). Can a u280 GRU fit a decoder better than MWPM on data
# it sees many times? d=9 and d=7, each trained on shots [0, 2,000,000) of its FORMAL pool only
# (no extension pools), peak LR 1e-3, cosine to 3e-5, 3-epoch warmup, Adam global_clipnorm 0.64
# (as 9-1), batch 10,000, 300 epochs (200 updates per epoch, 60,000 in total), no other
# regularization, seed 0, reset_after=False. Validation [15.0M, 15.2M) is still tracked by
# the launcher (it also scores the val-chosen checkpoint on the evaluation block).
# Then both the last-epoch weights and the val-chosen best weights are scored on the same 2M
# training shots, with MWPM decoded on them and McNemar (eval_student_on_tail.py).
# Also: L1 (d7 u280 100M, the 100M counterpart of R01) scored on its own training slice [0, 2M).
# The sealed block is not read.
#
#     bash run.sh      (on grace1, from this folder; runs the three parts one after another)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../../src" && pwd)"
IMG=qdec:tf215
pod () {  # pod <container name> <command...>
  local n="$1"; shift
  podman run --rm --name "$n" --security-opt=label=disable --device nvidia.com/gpu=all \
    -v "$HOME:$HOME" -e HOME="$HOME" -e TF_DETERMINISTIC_OPS=1 -e TF_CUDNN_DETERMINISTIC=1 \
    -e MALLOC_ARENA_MAX=2 -w "$SRC" "$IMG" "$@"
}
score () {  # score <name> <d> <weights> -> one row in train_slice_scores.csv
  pod "ovf_eval_$1" python3 eval_student_on_tail.py --student gru --inputs evts --hidden --units 280 \
    --d "$2" --p 0.004 --rounds "$2" --pool "$HOME/pools_d$2_p004_19M/data_d$2_p0.004_r$2_FORMAL.npz" \
    --weights "$3" --eval-start 0 --n-test 2000000 --batch-size 10000 --no-tf32 --gpu-mem-mib 4000 \
    --out-csv "$HERE/train_slice_scores.csv" > "$HERE/score_$1.log" 2>&1
  echo "[overfit] scored $1 exit $?"
}
train () {  # train <name> <d>
  local OUT="$HERE/$1"
  mkdir -p "$OUT"
  pod "ovf_$1" env D="$2" ROUNDS="$2" SEEDS=0 UNITS=280 P=0.004 \
      POOL="$HOME/pools_d$2_p004_19M/data_d$2_p0.004_r$2_FORMAL.npz" NTRAIN=2000000 BATCH=10000 \
      EPOCHS=300 LR=0.001 LR_SCHEDULE=cosine LR_END=3e-5 LR_WARMUP_EPOCHS=3 GLOBAL_CLIPNORM=0.64 \
      EARLY_STOP=0 RESET_AFTER=0 UNROLL=0 EVAL_NO_TF32=1 GPU_MEM_MIB=4000 MIN_FREE_MIB=4000 \
      OUT="$OUT" bash gh200_run_student_scaling.sh > "$OUT/podman.log" 2>&1
  echo "[overfit] trained $1 exit $?"
  score "${1}_last" "$2" "$(ls "$OUT"/seed0/ckpt/*.lastepoch.weights.h5)"
  score "${1}_best" "$2" "$(ls "$OUT"/seed0/ckpt/*.best.weights.h5)"
}
score L1_d7_100M 7 "$(ls $HOME/FNAL-AI4QC-GRU/experiments/scaling100M_2026-09-29/L1_d7_u280_100M_s0_*/seed0/ckpt/*.best.weights.h5)"
train d9_2M_300ep 9
train d7_2M_300ep 7
echo "[overfit] done"
