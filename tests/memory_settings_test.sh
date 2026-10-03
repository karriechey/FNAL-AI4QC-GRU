#!/usr/bin/env bash
# Created: 2026-10-01
# Last updated: 2026-10-01
#
# Host-memory settings test for training (requested 2026-10-01 before the d9 runs).
#
# The same job is trained for 3 epochs under several environment settings, and for each
# the peak host memory (RAM + swap of the training process) and the training/validation
# loss of every epoch are recorded. A setting is acceptable only if its losses match the
# baseline within the run-to-run noise measured by running the baseline twice (A0, A1).
#
# Job: d7 u280, 20M shots (FORMAL 15M + TRAINONLY seed43 5M), batch 10,000, cosine
# 3e-3 -> 3e-5 over the 3 epochs, global_clipnorm 0.64, seed 0, determinism on, 1.6 GB GPU
# cap. Same train_student.py call as the launcher, minus the evaluation-block scoring step.
#
# Variants (environment of the training process):
#   A0, A1  none (current settings), twice
#   M       MALLOC_ARENA_MAX=2                                 (glibc per-thread heaps)
#   P       HOSTDATA_MAP_PARALLEL=4 HOSTDATA_PREFETCH=4         (fixed tf.data depths)
#   MP      M + P
#   H       TF_GPU_HOST_MEM_LIMIT_IN_MB=8192                   (pinned host memory cap)
#   MPH     M + P + H
# Run in two waves of at most four jobs, loads started 60 s apart.
#
#     bash tests/memory_settings_test.sh     (from the repo root, on grace1)
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
O="$REPO/experiments/memory_settings_test/$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$O"
echo "repo_head $(git -C "$REPO" rev-parse HEAD) branch $(git -C "$REPO" branch --show-current 2>/dev/null)" > "$O/summary.txt"
P7="$HOME/pools_d7_p004_19M/data_d7_p0.004_r7"
declare -A ENVS=(
  [A0]="" [A1]=""
  [M]="MALLOC_ARENA_MAX=2"
  [P]="HOSTDATA_MAP_PARALLEL=4 HOSTDATA_PREFETCH=4"
  [MP]="MALLOC_ARENA_MAX=2 HOSTDATA_MAP_PARALLEL=4 HOSTDATA_PREFETCH=4"
  [H]="TF_GPU_HOST_MEM_LIMIT_IN_MB=8192"
  [MPH]="MALLOC_ARENA_MAX=2 HOSTDATA_MAP_PARALLEL=4 HOSTDATA_PREFETCH=4 TF_GPU_HOST_MEM_LIMIT_IN_MB=8192"
)
start () {  # start <variant>
  local v="$1" e envargs=()
  for e in ${ENVS[$v]}; do envargs+=(-e "$e"); done
  mkdir -p "$O/$v"
  echo "${ENVS[$v]:-none}" > "$O/$v/env.txt"
  podman run -d --rm --name "memtest_$v" --security-opt=label=disable --device nvidia.com/gpu=all \
    -v "$HOME:$HOME" -e HOME="$HOME" -e TF_DETERMINISTIC_OPS=1 -e TF_CUDNN_DETERMINISTIC=1 \
    "${envargs[@]}" -w "$REPO/src" qdec:tf215 \
    bash -c "python3 train_student.py --student gru --inputs evts --hidden --units 280 --gpu-mem-mib 1600 \
      --no-early-stopping --extra-train-pool ${P7}_TRAINONLY_seed43.npz --extra-train-n 5000000 \
      --d 7 --p 0.004 --rounds 7 --pool ${P7}_FORMAL.npz --n-train 15000000 \
      --val-start 15000000 --val-n 200000 --eval-start 15200000 --n-test 200000 \
      --alpha 1.0 --temperature 1.0 --batch-size 10000 --lr 0.003 --lr-schedule cosine --lr-end 3e-5 \
      --global-clipnorm 0.64 --epochs 3 --seed 0 --require-determinism \
      --ckpt-dir $O/$v/ckpt --out-dir $O/$v/runs --tag t --run-tag t > $O/$v/run.log 2>&1" > /dev/null
}
sample () {  # record max(RSS + swap) over the processes of each running memtest container
  local c p r s best
  for c in $(podman ps --format '{{.Names}}' | grep '^memtest_'); do
    best=0
    for p in $(podman top "$c" hpid 2>/dev/null | tail -n +2); do
      r=$(awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null); s=$(awk '/VmSwap/{print $2}' /proc/$p/status 2>/dev/null)
      [ -n "$r" ] && [ $(( r + ${s:-0} )) -gt $best ] && best=$(( r + ${s:-0} ))
    done
    echo -e "$(date -u +%H:%M:%S)\t${c#memtest_}\t$best\t$(grep -c 'ms/step' "$O/${c#memtest_}/run.log" 2>/dev/null)" >> "$O/memory.tsv"
  done
}
wave () {
  local v
  for v in "$@"; do start "$v"; for i in $(seq 1 6); do sample; sleep 10; done; done
  while podman ps --format '{{.Names}}' | grep -q '^memtest_'; do sample; sleep 10; done
}
wave A0 A1 M P
wave MP H MPH
podman run -i --rm --security-opt=label=disable -v "$HOME:$HOME" -e CUDA_VISIBLE_DEVICES= qdec:tf215 \
  python3 - "$O" <<'PY' 2>&1 | grep -v -e NVIDIA -e nvidia -e '^=' -e TensorFlow -e Copyright -e 'By pulling' -e '^$' | tee -a "$O/summary.txt"
import csv, json, os, sys
o = sys.argv[1]
peak, final = {}, {}
for t, v, kb, ep in csv.reader(open(f'{o}/memory.tsv'), delimiter='\t'):
    peak[v] = max(peak.get(v, 0), int(kb)); final[v] = int(kb)
def hist(v):
    p = f'{o}/{v}/runs/t.history.json'
    return json.load(open(p)) if os.path.exists(p) else None
base = hist('A0')
print(f"{'variant':8s} {'env':78s} {'peak GB':>8s} {'end GB':>7s}  max|diff| vs A0 (loss, val_loss, val_errors)")
for v in ['A0', 'A1', 'M', 'P', 'MP', 'H', 'MPH']:
    h = hist(v); env = open(f'{o}/{v}/env.txt').read().strip()
    if h is None or base is None:
        print(f'{v:8s} {env:78s} run did not finish; see {v}/run.log'); continue
    d = [max(abs(a - b) for a, b in zip(h[k], base[k])) for k in ('loss', 'val_loss', 'val_errors')]
    print(f"{v:8s} {env:78s} {peak.get(v, 0) / 1048576:8.1f} {final.get(v, 0) / 1048576:7.1f}  "
          f"{d[0]:.3g}, {d[1]:.3g}, {d[2]:.0f}   epochs={len(h['loss'])}")
PY
