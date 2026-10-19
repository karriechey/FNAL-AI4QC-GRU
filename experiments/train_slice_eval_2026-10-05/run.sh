#!/usr/bin/env bash
# Created: 2026-10-05
# Last modified: 2026-10-05
#
# Evaluation only (no training): the best checkpoints of the best d=7 run (R01, u280, 200M)
# and the best d=9 run (9-1, u280, 100M) scored on shots [0, 2,000,000) of each run's own
# primary FORMAL pool. Those shots are inside the training prefix [0, 15M) both runs trained
# on, so the comparison with the evaluation block [15.2M, 17M) shows how much of the gap to
# MWPM is generalization. MWPM is decoded on the same 2M shots and paired shot by shot
# (eval_student_on_tail.py). Same settings as the evaluation-block scoring: TF32 off, batch
# 10,000, reset_after=False, the .best checkpoint. The sealed block is not read.
#
#     bash run.sh      (on grace1, from this folder)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../../src" && pwd)"
R01=$(ls $HOME/FNAL-AI4QC-GRU-compact/experiments/queue12_2026-09-30/R01_d7_u280_200M_2*/seed0/ckpt/*.best.weights.h5)
R91=$(ls $HOME/FNAL-AI4QC-GRU-mwpm/experiments/mwpm_fixes_d9_2026-10/9-1_*/seed0/ckpt/*.best.weights.h5)
run () {  # run <name> <d> <weights>
  local name="$1" d="$2" w="$3"
  podman run --rm --name "tse_$name" --security-opt=label=disable --device nvidia.com/gpu=all \
    -v "$HOME:$HOME" -e HOME="$HOME" -e TF_DETERMINISTIC_OPS=1 -e TF_CUDNN_DETERMINISTIC=1 \
    -e MALLOC_ARENA_MAX=2 -w "$SRC" qdec:tf215 \
    python3 eval_student_on_tail.py --student gru --inputs evts --hidden --units 280 \
      --d "$d" --p 0.004 --rounds "$d" \
      --pool "$HOME/pools_d${d}_p004_19M/data_d${d}_p0.004_r${d}_FORMAL.npz" \
      --weights "$w" --eval-start 0 --n-test 2000000 --batch-size 10000 \
      --no-tf32 --gpu-mem-mib 4000 \
      --out-csv "$HERE/train_slice_eval.csv" > "$HERE/${name}.log" 2>&1
  echo "[train-slice] $name exit $?"
}
run R01_d7 7 "$R01"
run 9-1_d9 9 "$R91"
echo "[train-slice] done"
