#!/usr/bin/env bash
# Created: 2026-09-30
# Last updated: 2026-09-30
#
# Check the compact training-only pool format against the full one, before any compact
# pool is used for training.
#
# For d = 5, 7, 9 at p = 0.004 (r = d): write the same 30,000 shots twice with
# gen_train_extension.py and the same seed, once full (measurements, det_evts, flips) and
# once --compact (bit-packed detection events, flips). Then check that
#   1. the flips are identical and the unpacked detection events equal the full pool's;
#   2. train_student.load_extension_features (the loader training uses) returns identical
#      feature tensors (same dtype, shape and every value) and identical labels for both
#      files, for the whole pool and for a prefix (take < n).
# CPU only, inside the qdec:tf215 image. Test pools use seed 9001 and live in this test's
# output folder only.
#
#     bash tests/compact_pool_check.sh     (from the repo root, on grace1)
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
O="$REPO/experiments/compact_pool_check/$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$O"
echo "repo_head $(git -C "$REPO" rev-parse HEAD) branch $(git -C "$REPO" branch --show-current)" > "$O/summary.txt"
RUN=(podman run -i --rm --security-opt=label=disable -v "$HOME:$HOME" -e CUDA_VISIBLE_DEVICES= -w "$REPO/src" qdec:tf215)
for d in 5 7 9; do
  "${RUN[@]}" python3 gen_train_extension.py --d $d --p 0.004 --rounds $d --n 30000 --gen-seed 9001 --out-dir "$O/full" >> "$O/gen.log" 2>&1
  "${RUN[@]}" python3 gen_train_extension.py --d $d --p 0.004 --rounds $d --n 30000 --gen-seed 9001 --out-dir "$O/compact" --compact >> "$O/gen.log" 2>&1
done
"${RUN[@]}" python3 - "$O" <<'PY' 2>&1 | grep -v -e NVIDIA -e nvidia -e '^=' -e TensorFlow -e Copyright -e 'By pulling' -e '^$' | tee -a "$O/summary.txt"
import os, sys, numpy as np
from train_student import load_extension_features
from types_cfg import get_types
o = sys.argv[1]
ok = True
for d in (5, 7, 9):
    name = f'data_d{d}_p0.004_r{d}_TRAINONLY_seed9001.npz'
    zf, zc = np.load(f'{o}/full/{name}'), np.load(f'{o}/compact/{name}')
    same_flips = np.array_equal(zf['flips'], zc['flips'])
    unpacked = np.unpackbits(zc['det_evts_packed'], axis=1, count=int(zc['n_det']))
    same_det = np.array_equal(unpacked.astype(np.int8), zf['det_evts'])
    binary_t, _t, idx_t, _p = get_types(d, d, 3)
    res = []
    for take in (30000, 12345):
        xf, ff = load_extension_features(zf, take, d, d, 0.004, 'evts', 'gru', idx_t, binary_t)
        xc, fc = load_extension_features(zc, take, d, d, 0.004, 'evts', 'gru', idx_t, binary_t)
        same = (xf.dtype == xc.dtype and xf.shape == xc.shape and np.array_equal(xf, xc)
                and ff.dtype == fc.dtype and np.array_equal(ff, fc))
        res.append((take, xf.shape, str(xf.dtype), same))
    sz = (os.path.getsize(f'{o}/full/{name}'), os.path.getsize(f'{o}/compact/{name}'))
    this = same_flips and same_det and all(r[3] for r in res)
    ok &= this
    print(f'd={d}: flips identical {same_flips}; unpacked det_evts == full det_evts {same_det}; '
          + '; '.join(f'take {t}: features {s} {dt} identical {q}' for t, s, dt, q in res)
          + f'; file bytes full {sz[0]:,} compact {sz[1]:,} ({sz[0] / sz[1]:.1f}x smaller)')
print('COMPACT POOL CHECK:', 'PASS' if ok else 'FAIL')
PY
