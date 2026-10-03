#!/usr/bin/env bash
# Created: 2026-10-01
# Last updated: 2026-10-01
#
# Per-epoch host-memory growth during training (found 2026-10-01: the queue's jobs grew
# 0.5-6 GB per epoch until grace1 swapped). Same job twice, 10 epochs each, memory (RAM +
# swap of the training process) recorded at every epoch end:
#   A  current settings
#   M  MALLOC_ARENA_MAX=2
# Job: d7 u280, 50M (FORMAL 15M + seeds 43, 44), batch 10,000, cosine 3e-3 -> 3e-5 over the
# 10 epochs, global_clipnorm 0.64, seed 0, determinism on, 1.6 GB GPU cap. Also checks that
# A and M give identical losses.
#
#     bash tests/memory_growth_test.sh     (from the repo root of the compact-pools checkout)
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
O="$REPO/experiments/memory_growth_test/$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$O"
echo "code $(git -C "$REPO" rev-parse HEAD)" > "$O/summary.txt"
P7="$HOME/pools_d7_p004_19M/data_d7_p0.004_r7"
start () { local v="$1"; shift; mkdir -p "$O/$v"
  podman run -d --rm --name "mgtest_$v" --security-opt=label=disable --device nvidia.com/gpu=all \
    -v "$HOME:$HOME" -e HOME="$HOME" -e TF_DETERMINISTIC_OPS=1 -e TF_CUDNN_DETERMINISTIC=1 "$@" -w "$REPO/src" qdec:tf215 \
    bash -c "python3 train_student.py --student gru --inputs evts --hidden --units 280 --gpu-mem-mib 1600 \
      --no-early-stopping --extra-train-pool ${P7}_TRAINONLY_seed43.npz ${P7}_TRAINONLY_seed44.npz --extra-train-n 5000000 30000000 \
      --d 7 --p 0.004 --rounds 7 --pool ${P7}_FORMAL.npz --n-train 15000000 \
      --val-start 15000000 --val-n 200000 --eval-start 15200000 --n-test 200000 \
      --alpha 1.0 --temperature 1.0 --batch-size 10000 --lr 0.003 --lr-schedule cosine --lr-end 3e-5 \
      --global-clipnorm 0.64 --epochs 10 --seed 0 --require-determinism \
      --ckpt-dir $O/$v/ckpt --out-dir $O/$v/runs --tag t --run-tag t > $O/$v/run.log 2>&1" > /dev/null; }
start A
sleep 240
start M -e MALLOC_ARENA_MAX=2
while podman ps --format '{{.Names}}' | grep -q '^mgtest_'; do
  for c in $(podman ps --format '{{.Names}}' | grep '^mgtest_'); do
    v=${c#mgtest_}; best=0
    for p in $(podman top "$c" hpid 2>/dev/null | tail -n +2); do
      r=$(awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null); s=$(awk '/VmSwap/{print $2}' /proc/$p/status 2>/dev/null)
      [ -n "$r" ] && [ $(( r + ${s:-0} )) -gt $best ] && best=$(( r + ${s:-0} ))
    done
    echo -e "$(date -u +%H:%M:%S)\t$v\t$(grep -c 'ms/step' $O/$v/run.log)\t$best" >> "$O/memory.tsv"
  done
  sleep 15
done
podman run -i --rm --security-opt=label=disable -v "$HOME:$HOME" -e CUDA_VISIBLE_DEVICES= qdec:tf215 python3 - "$O" <<'PY' 2>&1 | grep -v -e NVIDIA -e nvidia -e '^=' -e TensorFlow -e Copyright -e 'By pulling' -e '^$' | tee -a "$O/summary.txt"
import csv, json, sys
o = sys.argv[1]
mx = {}
for t, v, ep, kb in csv.reader(open(f'{o}/memory.tsv'), delimiter='\t'):
    mx.setdefault(v, {}); e = int(ep); mx[v][e] = max(mx[v].get(e, 0), int(kb))
for v in ('A', 'M'):
    eps = sorted(mx.get(v, {}))
    print(v, 'max GB while in epoch (completed epochs: GB):', ' '.join(f'{e}:{mx[v][e]/1048576:.1f}' for e in eps))
    if len(eps) > 4:
        a, b = eps[2], eps[-1]
        print(f'   growth from {a} to {b} completed epochs: {(mx[v][b]-mx[v][a])/1048576/(b-a):.2f} GB per epoch')
try:
    ha = json.load(open(f'{o}/A/runs/t.history.json')); hm = json.load(open(f'{o}/M/runs/t.history.json'))
    print('M vs A max |diff| loss, val_loss:', max(abs(x-y) for x, y in zip(ha['loss'], hm['loss'])), max(abs(x-y) for x, y in zip(ha['val_loss'], hm['val_loss'])))
except Exception as ex:
    print('history compare failed:', ex)
PY
