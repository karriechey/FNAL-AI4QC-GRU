#!/usr/bin/env python3
# from FNAL-QCDecoding-FPGA/mwpm_on_block.py @ d5ee86f
# Created: 2026-08-19
# Last modified: 2026-10-06 (only the block's rows are read, via pool_rows; --flips-sha streams the whole flips array and refuses a pool with a sealed block).
# Earlier: 2026-09-25 (build_circuit from circuit_generators)

"""Decode MWPM on one shot block of a pool and record the baseline.

Writes a JSON (p_L, residuals, standard error, block, pool flips SHA) and, optionally, per-shot
correctness for paired McNemar tests. No TensorFlow; runs in ~/tfenv on grace1.

python mwpm_on_block.py --pool <pool.npz> --d 7 --p 0.004 --rounds 7 \
    --start 15200000 --n 1800000 \
    --out-json mwpm_eval_block_d7_r7_p004.json \
    --dump-per-shot mwpm_eval_block_d7_r7_p004_per_shot.npz
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

from circuit_generators import build_circuit   # shared noise-model definition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True, help='pool npz holding det_evts and flips')
    ap.add_argument('--d', type=int, required=True)
    ap.add_argument('--p', type=float, required=True)
    ap.add_argument('--rounds', type=int, required=True)
    ap.add_argument('--start', type=int, required=True,
                    help='first shot index of the block, absolute in the pool')
    ap.add_argument('--n', type=int, required=True, help='number of shots in the block')
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--dump-per-shot', default=None,
                    help='optional npz of per-shot truth/prediction/correctness')
    ap.add_argument('--flips-sha', action='store_true',
                    help='hash the pool-wide flips array (reads the whole array; use it '
                         'once at pool acceptance, skip it on repeat measurements)')
    args = ap.parse_args()

    if not os.path.exists(args.pool):
        raise SystemExit(f"[mwpm] missing pool {args.pool}")

    import pool_rows as PR
    N = PR.n_rows(args.pool, 'det_evts')            # header only
    lo, hi = args.start, args.start + args.n
    if hi > N:
        raise SystemExit(f"[mwpm] block [{lo:,}, {hi:,}) exceeds pool size {N:,}")
    print(f"[mwpm] pool {args.pool}  shots={N:,}", flush=True)
    print(f"[mwpm] block [{lo:,}, {hi:,})  d={args.d} r={args.rounds} p={args.p}",
          flush=True)

    det = PR.rows(args.pool, 'det_evts', lo, hi).astype(np.int8)
    truth = PR.rows(args.pool, 'flips', lo, hi).astype(np.int8).reshape(-1)

    flips_sha = ''   # hash of the whole flips array, pins the baseline to this pool
    if args.flips_sha:
        t = time.time()
        import partitions as P
        if P.sealed_block(args.pool, N)[0] is not None:
            raise SystemExit("[mwpm] --flips-sha reads every row, the sealed rows too; the "
                             "pool's .fingerprint.json already holds its flips SHA-256. STOP.")
        h = hashlib.sha256()
        for a in range(0, N, 10_000_000):
            h.update(np.ascontiguousarray(PR.rows(args.pool, 'flips', a, min(a + 10_000_000, N))).tobytes())
        flips_sha = h.hexdigest()
        print(f"[mwpm] flips sha256 {flips_sha[:16]} ({time.time() - t:.0f}s)", flush=True)

    import pymatching
    import stim
    circ = build_circuit(args.d, args.p, args.rounds)
    if circ.num_detectors != det.shape[1]:
        raise SystemExit(f"[mwpm] circuit has {circ.num_detectors} detectors, the pool "
                         f"stores {det.shape[1]} -- these are not the same experiment.")
    dem = circ.detector_error_model(decompose_errors=True)
    pym = pymatching.Matching.from_detector_error_model(dem)

    t0 = time.time()
    pred = pym.decode_batch(det, bit_packed_predictions=False,
                            bit_packed_shots=False).astype(np.int8).reshape(-1)
    decode_s = time.time() - t0

    correct = (pred == truth)
    resid = int((~correct).sum())
    p_L = resid / args.n
    se = float(np.sqrt(p_L * (1 - p_L) / args.n))   # binomial floor on the uncertainty
    base_rate = float(truth.mean())

    print(f"[mwpm] p_L={p_L:.6f} +/- {se:.6f}  ({resid:,} residual errors of "
          f"{args.n:,})  base_rate={base_rate:.5f}  decode {decode_s / 60:.1f} min",
          flush=True)
    if resid < 500:
        print("[mwpm] warning: fewer than 500 residual errors -- this estimate is "
              "count-limited; a larger block is needed to resolve a ratio.", flush=True)

    meta = dict(pool=os.path.abspath(args.pool), pool_shots=int(N),
                d=args.d, p=args.p, rounds=args.rounds,
                block_start=lo, block_stop=hi, n=args.n,
                mwpm_p_L=p_L, mwpm_se=se, residual_errors=resid,
                base_rate=base_rate, decode_seconds=round(decode_s, 1),
                flips_sha256=flips_sha,
                stim_version=stim.__version__,
                pymatching_version=getattr(pymatching, '__version__', 'unknown'),
                numpy_version=np.__version__,
                measured_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    with open(args.out_json, 'w') as fh:
        json.dump(meta, fh, indent=2)
    print(f"[mwpm] wrote {args.out_json}", flush=True)

    if args.dump_per_shot:
        np.savez_compressed(args.dump_per_shot,
                            shot_idx=np.arange(lo, hi, dtype=np.int64),
                            truth=truth, mwpm_pred=pred, mwpm_correct=correct,
                            d=args.d, p=args.p, rounds=args.rounds)
        print(f"[mwpm] wrote {args.dump_per_shot}", flush=True)


if __name__ == '__main__':
    main()
