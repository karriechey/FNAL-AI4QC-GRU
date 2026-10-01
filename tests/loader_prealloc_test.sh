#!/usr/bin/env bash
# Created: 2026-10-01
# Last updated: 2026-10-01
#
# Check the preallocated, chunked training-set loader (compact-pools branch, 2026-10-01)
# against the previous loader, before the d9 runs use it.
#
#   OLD20  previous loader (main checkout ~/FNAL-AI4QC-GRU/src), d7 u280 20M, 3 epochs
#   NEW20  new loader (this checkout), same job
#   NEW100 new loader, d7 u280 100M (FORMAL 15M + seeds 43/44/45), 3 epochs, memory only
#
# Same training call as tests/memory_settings_test.sh (seed 0, determinism on, cosine over
# the 3 epochs, global_clipnorm 0.64, 1.6 GB GPU cap). PASS for the loader when NEW20's
# per-epoch loss, val_loss and val_errors equal OLD20's exactly. Peak host memory (RAM +
# swap of the training process) is recorded for all three. Jobs start one at a time.
#
#     bash tests/loader_prealloc_test.sh     (from the repo root of the compact-pools checkout)
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OLDSRC="$HOME/FNAL-AI4QC-GRU/src"
O="$REPO/experiments/loader_prealloc_test/$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$O"
{ echo "new code $(git -C "$REPO" rev-parse HEAD)"; echo "old code $(git -C "$HOME/FNAL-AI4QC-GRU" rev-parse HEAD)"; } > "$O/summary.txt"
P7="$HOME/pools_d7_p004_19M/data_d7_p0.004_r7"
run () {  # run <name> <src dir> <extra pools> <extra counts>
  local v="$1" src="$2" xp="$3" xn="$4"; mkdir -p "$O/$v"
  podman run -d --rm --name "ldtest_$v" --security-opt=label=disable --device nvidia.com/gpu=all \
    -v "$HOME:$HOME" -e HOME="$HOME" -e TF_DETERMINISTIC_OPS=1 -e TF_CUDNN_DETERMINISTIC=1 -w "$src" qdec:tf215 \
    bash -c "python3 train_student.py --student gru --inputs evts --hidden --units 280 --gpu-mem-mib 1600 \
      --no-early-stopping --extra-train-pool $xp --extra-train-n $xn \
      --d 7 --p 0.004 --rounds 7 --pool ${P7}_FORMAL.npz --n-train 15000000 \
      --val-start 15000000 --val-n 200000 --eval-start 15200000 --n-test 200000 \
      --alpha 1.0 --temperature 1.0 --batch-size 10000 --lr 0.003 --lr-schedule cosine --lr-end 3e-5 \
      --global-clipnorm 0.64 --epochs 3 --seed 0 --require-determinism \
      --ckpt-dir $O/$v/ckpt --out-dir $O/$v/runs --tag t --run-tag t > $O/$v/run.log 2>&1" > /dev/null
  local best=0 p r s t
  while podman ps --format '{{.Names}}' | grep -q "^ldtest_$v$"; do
    for p in $(podman top "ldtest_$v" hpid 2>/dev/null | tail -n +2); do
      r=$(awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null); s=$(awk '/VmSwap/{print $2}' /proc/$p/status 2>/dev/null)
      [ -n "$r" ] && t=$(( r + ${s:-0} )) && [ $t -gt $best ] && best=$t
    done
    sleep 10
  done
  echo "$best" > "$O/$v/peak_kb"
}
run OLD20 "$OLDSRC" "${P7}_TRAINONLY_seed43.npz" "5000000" &
sleep 90
run NEW20 "$REPO/src" "${P7}_TRAINONLY_seed43.npz" "5000000" &
wait
run NEW100 "$REPO/src" "${P7}_TRAINONLY_seed43.npz ${P7}_TRAINONLY_seed44.npz ${P7}_TRAINONLY_seed45.npz" "5000000 30000000 50000000"
podman run -i --rm --security-opt=label=disable -v "$HOME:$HOME" -e CUDA_VISIBLE_DEVICES= qdec:tf215 \
  python3 - "$O" <<'PY' 2>&1 | grep -v -e NVIDIA -e nvidia -e '^=' -e TensorFlow -e Copyright -e 'By pulling' -e '^$' | tee -a "$O/summary.txt"
import json, os, sys
o = sys.argv[1]
H = {v: json.load(open(f'{o}/{v}/runs/t.history.json')) for v in ('OLD20', 'NEW20', 'NEW100')
     if os.path.exists(f'{o}/{v}/runs/t.history.json')}
for v in ('OLD20', 'NEW20', 'NEW100'):
    pk = int(open(f'{o}/{v}/peak_kb').read()) / 1048576 if os.path.exists(f'{o}/{v}/peak_kb') else float('nan')
    print(f'{v:7s} peak host memory (RAM+swap) {pk:6.1f} GB  epochs {len(H[v]["loss"]) if v in H else "-"}')
if 'OLD20' in H and 'NEW20' in H:
    d = {k: max(abs(a - b) for a, b in zip(H['OLD20'][k], H['NEW20'][k])) for k in ('loss', 'val_loss', 'val_errors')}
    print('NEW20 vs OLD20 max |diff|:', d)
    print('LOADER CHECK:', 'PASS' if all(x == 0 for x in d.values()) else 'FAIL')
PY
