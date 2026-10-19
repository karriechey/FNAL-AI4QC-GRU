#!/usr/bin/env bash
# Created: 2026-10-14
# Last modified: 2026-10-14
# Waits for a run's container to stop, then scores its checkpoints from epoch 40 on (the epoch
# snapshots plus .best.weights.h5) on the selection block.
#   bash score_after_training.sh <run folder> <model> <d> <p> <units...>
set -uo pipefail
cd "$(dirname "$0")"
run=$1 model=$2 d=$3 p=$4; shift 4
arm=$(basename "$run"); arm=${arm%%_u*}
while podman ps --format '{{.Names}}' | grep -qx "probe_$arm"; do sleep 120; done
sleep 60                                     # snapshot_epochs.py's last pass
pt=$(printf '%03d' "$(python3 -c "print(round($p*1000))")")
pool=$(ls ~/pools_d${d}_p${pt}_19M/data_d${d}_p${p}_r${d}_FORMAL*.npz)
w=()
for f in "$run"/seed*/epochs/epoch*.weights.h5 "$run"/seed*/ckpt/*.best.weights.h5; do
  e=$(python3 -c "import json,glob,os,re,sys
f=sys.argv[1]
m=re.search(r'epoch(\d+)\.weights', f)
print(m.group(1) if m else json.load(open(glob.glob(os.path.join(os.path.dirname(f), '*.best.json'))[0]))['best_epoch'])" "$f")
  [ "$((10#$e))" -ge 40 ] && w+=("$f")
done
podman run --rm --name "sel_$model" --security-opt=label=disable --device nvidia.com/gpu=all \
  -v "$HOME:$HOME" -e HOME="$HOME" -w "$PWD" qdec:tf215 python3 score_selection.py \
  --model "$model" --d "$d" --p "$p" --units "$@" --pool "$pool" --steps-per-epoch 10000 \
  --out-csv selection_scores.csv "${w[@]}"
rc=$?
echo "$(date -u +%FT%TZ) $model scored ${#w[@]} checkpoints, exit $rc"
