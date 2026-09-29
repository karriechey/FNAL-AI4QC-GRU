#!/usr/bin/env bash
# Created: 2026-09-29
# Last updated: 2026-09-29
#
# Check that resuming a run (train_student.py --resume-weights/--initial-epoch/
# --resume-best-json/--resume-history) reproduces an uninterrupted run exactly.
#
#   A  : 4 epochs straight through
#   B1 : 2 epochs;  B2 : B1 resumed to 4 epochs
#
# Small d5 GRU (u32), 200k training shots, batch 10,000, determinism on. DEVICE=gpu (default)
# or DEVICE=cpu. PASS when
# A and B2 have identical per-epoch history, identical final weights and optimizer state,
# and the same best-checkpoint record.
#
#     bash tests/resume_exactness.sh      (from the repo root, on the GPU host)
set -uo pipefail
REPO="$HOME/FNAL-AI4QC-GRU"
DEVICE="${DEVICE:-gpu}"
O="$REPO/experiments/resume_exactness/$(date -u +%Y%m%dT%H%M%SZ)_$DEVICE"; mkdir -p "$O"
if [ "$DEVICE" = cpu ]; then DEV=(-e CUDA_VISIBLE_DEVICES=); GPUFLAG=""; else DEV=(--device nvidia.com/gpu=all); GPUFLAG="--gpu-mem-mib 1600"; fi
P5="$HOME/pools_d5_p004_19M/data_d5_p0.004_r5_FORMAL.npz"
T=t
COMMON="--student gru --inputs evts --hidden --units 32 --d 5 --p 0.004 --rounds 5 --pool $P5 \
 --n-train 200000 --val-start 15000000 --val-n 20000 --eval-start 15200000 --n-test 20000 \
 --alpha 1.0 --temperature 1.0 --batch-size 10000 --lr 0.003 --lr-schedule constant \
 --no-early-stopping --seed 0 --require-determinism $GPUFLAG --tag $T --run-tag $T"
run () {  # run <name> <extra args...>
  local n="$1"; shift
  podman run --rm --security-opt=label=disable "${DEV[@]}" -v "$HOME:$HOME" \
    -e HOME="$HOME" -e TF_DETERMINISTIC_OPS=1 -e TF_CUDNN_DETERMINISTIC=1 -w "$REPO/src" qdec:tf215 \
    python3 train_student.py $COMMON --ckpt-dir "$O/$n/ckpt" --out-dir "$O/$n/runs" "$@" \
    > "$O/$n.log" 2>&1
  echo "$n exit $?" | tee -a "$O/summary.txt"
}
echo "repo_head $(git -C "$REPO" rev-parse HEAD)" > "$O/summary.txt"
run A --epochs 4
run B1 --epochs 2
mkdir -p "$O/B2/ckpt"; cp "$O"/B1/ckpt/$T.best.* "$O/B2/ckpt/"
run B2 --epochs 4 --resume-weights "$O/B1/ckpt/$T.lastepoch.weights.h5" --initial-epoch 2 \
  --resume-best-json "$O/B2/ckpt/$T.best.json" --resume-history "$O/B1/runs/$T.history.json"
podman run -i --rm --security-opt=label=disable -v "$HOME:$HOME" -e CUDA_VISIBLE_DEVICES= qdec:tf215 \
  python3 - "$O" "$T" <<'PY' 2>&1 | grep -v -e NVIDIA -e nvidia -e '^=' -e TensorFlow -e Copyright -e 'By pulling' -e '^$' | tee -a "$O/summary.txt"
import json, sys, h5py, numpy as np
o, t = sys.argv[1], sys.argv[2]
ok = True
ha = json.load(open(f'{o}/A/runs/{t}.history.json')); hb = json.load(open(f'{o}/B2/runs/{t}.history.json'))
for k in sorted(ha):
    same = ha[k] == hb.get(k)
    ok &= same
    print(f'history {k:18s} {"identical" if same else "DIFFERENT"}  A={ha[k]}  B2={hb.get(k)}')
def arrays(p):
    out = {}
    with h5py.File(p, 'r') as f:
        f.visititems(lambda n, x: out.__setitem__(n, x[()]) if isinstance(x, h5py.Dataset) else None)
    return out
wa, wb = arrays(f'{o}/A/ckpt/{t}.lastepoch.weights.h5'), arrays(f'{o}/B2/ckpt/{t}.lastepoch.weights.h5')
same = wa.keys() == wb.keys() and all(np.array_equal(wa[k], wb[k]) for k in wa)
ok &= same
print(f'final weights + optimizer state ({len(wa)} arrays, incl. {sum("optimizer" in k for k in wa)} optimizer): {"identical" if same else "DIFFERENT"}')
ba, bb = json.load(open(f'{o}/A/ckpt/{t}.best.json')), json.load(open(f'{o}/B2/ckpt/{t}.best.json'))
ok &= ba == bb
print(f'best.json A={ba}  B2={bb}  {"identical" if ba == bb else "DIFFERENT"}')
print('RESUME EXACTNESS:', 'PASS' if ok else 'FAIL')
PY
