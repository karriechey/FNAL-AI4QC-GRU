#!/usr/bin/env bash
# Created: 2026-10-14
# Last modified: 2026-10-14
# selection-block scores for the finished p=0.004 fresh-shot runs (best + last epoch are all that was saved)
cd ~/FNAL-AI4QC-GRU-fresh2/experiments/selection_2026-10-14
F2=~/FNAL-AI4QC-GRU-fresh2/experiments/ceiling_probe_d7_2026-10-05
F1=~/FNAL-AI4QC-GRU-fresh/experiments/ceiling_probe_d7_2026-10-05
run() {  # model d units... -- ckptdir
  local model=$1 d=$2 ck=$3; shift 3
  podman run --rm --name sel_$model --security-opt=label=disable --device nvidia.com/gpu=all \
    -v "$HOME:$HOME" -e HOME="$HOME" -w "$PWD" qdec:tf215 python3 score_selection.py \
    --model $model --d $d --p 0.004 --units "$@" --pool ~/pools_d${d}_p004_19M/data_d${d}_p0.004_r${d}_FORMAL.npz \
    --steps-per-epoch 10000 --out-csv selection_scores.csv $ck/*.best.weights.h5 $ck/*.lastepoch.weights.h5 2>&1 | grep "epoch "
}
run f7w_p004 7 $F2/f7w_p004_u280x280_20261014T035645Z/seed0/ckpt 280 280
run f3280 9 $F1/f3280_u280x280x280_20261010T032442Z/seed0/ckpt 280 280 280
run f3280e240 9 $F2/f3280e240_u280x280x280_20261011T064457Z/seed0/ckpt 280 280 280
run f3280s1 9 $F2/f3280s1_u280x280x280_20261013T140127Z/seed1/ckpt 280 280 280
echo ALLDONE
