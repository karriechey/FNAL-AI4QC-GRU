#!/usr/bin/env bash
# Created: 2026-10-15
# Last modified: 2026-10-15
# Evaluation block FORMAL [15.2M, 17M) for the checkpoints the selection rule picked.
#   bash eval_selected.sh <model> <d> <p> <weights> <units...>
set -uo pipefail
cd "$(dirname "$0")"
model=$1 d=$2 p=$3 w=$4; shift 4
pt=$(printf '%03d' "$(python3 -c "print(round($p*1000))")")
pool=$(ls ~/pools_d${d}_p${pt}_19M/data_d${d}_p${p}_r${d}_FORMAL*.npz)
podman run --rm --name "evalsel_$model" --security-opt=label=disable --device nvidia.com/gpu=all \
  -v "$HOME:$HOME" -e HOME="$HOME" -e TF_DETERMINISTIC_OPS=1 -e TF_CUDNN_DETERMINISTIC=1 \
  -w "$HOME/FNAL-AI4QC-GRU-fresh2/src" qdec:tf215 python3 eval_student_on_tail.py --student gru \
  --inputs evts --hidden --units "$@" --d $d --p $p --rounds $d --pool "$pool" --weights "$w" \
  --eval-start 15200000 --n-test 1800000 --batch-size 10000 --no-tf32 --gpu-mem-mib 8000 \
  --out-csv "$PWD/eval_selected.csv" 2>&1 | grep -E "student p_L|mcnemar\]   (student-only|MWPM-only)|p_exact"
echo "$model done"
