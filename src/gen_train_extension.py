#!/usr/bin/env python3
# from FNAL-QCDecoding-FPGA/gen_train_extension.py @ d5ee86f
# Created: 2026-08-21
# Last modified: 2026-09-30 (--compact: detection events only, bit-packed; --formal: a primary pool with the FORMAL partitions, for new p values).
# Earlier 2026-09-25 (build_circuit from circuit_generators)

"""Generate a training-only pool: every shot is training data, no reserved blocks.

The seed must differ from the primary pool's; train_student.py refuses a matching seed.
--compact stores the same shots as bit-packed detection events + flips (about 1/16 the size).
--formal writes a primary FORMAL pool instead; --n must equal n_total (19,000,000).
Writes <pool>.npz and <pool>.fingerprint.json.

python gen_train_extension.py --d 5 --p 0.004 --rounds 5 --n 5000000 --gen-seed 43 --out-dir ~/pools_d5_p004_19M
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

from circuit_generators import build_circuit
from generate_pools import sample_pool, sample_pool_compact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, required=True)
    ap.add_argument('--p', type=float, required=True)
    ap.add_argument('--rounds', type=int, required=True)
    ap.add_argument('--n', type=int, required=True, help='shots, all of them training')
    ap.add_argument('--gen-seed', type=int, required=True,
                    help='must differ from the primary pool it augments')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--tag', default='TRAINONLY')
    ap.add_argument('--compact', action='store_true',
                    help='store bit-packed detection events + flips only (no measurements)')
    ap.add_argument('--formal', action='store_true',
                    help='write a primary FORMAL pool (full format) with the partitions of '
                         'configs/partitions.yaml')
    args = ap.parse_args()
    if args.formal and args.compact:
        raise SystemExit('[ext] --formal pools keep the full format; drop --compact')
    if args.formal:
        from partitions import load as load_partitions
        pcfg = load_partitions()['formal']
        if args.n != int(pcfg['n_total']):
            raise SystemExit(f"[ext] --formal needs --n {pcfg['n_total']} (the FORMAL layout)")
        args.tag = 'FORMAL'

    d, p, r = args.d, args.p, args.rounds
    os.makedirs(args.out_dir, exist_ok=True)
    fn = os.path.join(args.out_dir, f'data_d{d}_p{p:.3f}_r{r}_{args.tag}_seed{args.gen_seed}.npz')
    if os.path.exists(fn):
        raise SystemExit(f"[ext] {fn} exists -- refusing to overwrite.")

    circ = build_circuit(d, p, r)
    print(f"[ext] d={d} p={p} rounds={r}: {circ.num_detectors} detectors, "
          f"{r * (d ** 2 - 1) + d ** 2} measurements, gen_seed={args.gen_seed}")

    t0 = time.time()
    if args.compact:
        det_packed, n_det, flips = sample_pool_compact(circ, args.n, args.gen_seed)
        print(f"[ext] sampled {args.n:,} shots in {(time.time() - t0) / 60:.1f} min  "
              f"det_evts_packed{det_packed.shape} (n_det={n_det})", flush=True)
        np.savez(fn, det_evts_packed=det_packed, n_det=np.int64(n_det), flips=flips)
        meas_shape, det_shape = None, f'({args.n}, {n_det})'
    else:
        measurements, det_evts, flips = sample_pool(circ, args.n, args.gen_seed)
        print(f"[ext] sampled {args.n:,} shots in {(time.time() - t0) / 60:.1f} min  "
              f"meas{measurements.shape} det{det_evts.shape}", flush=True)
        np.savez(fn, measurements=measurements, det_evts=det_evts, flips=flips)
        meas_shape, det_shape = str(measurements.shape), str(det_evts.shape)
    size_gb = os.path.getsize(fn) / 1024 ** 3
    flips_sha = hashlib.sha256(np.ascontiguousarray(flips).tobytes()).hexdigest()

    meta = dict(path=os.path.abspath(fn), d=d, p=p, rounds=r, n_total=args.n,
                gen_seed=args.gen_seed, flips_sha256=flips_sha,
                purpose='training_only',
                partition_train=[0, args.n],
                partition_val=None, partition_test_SEALED=None,
                protocol='Training-only augmentation pool. Every shot is training data; '
                         'no validation, evaluation or sealed region exists here. The '
                         'primary pool supplies all three of those and is unmodified.',
                format=('compact: det_evts_packed (np.packbits axis=1), n_det, flips'
                        if args.compact else 'full: measurements, det_evts, flips'),
                measurements_shape=meas_shape,
                det_evts_shape=det_shape,
                base_rate=round(float(flips.mean()), 5),
                size_gb=round(size_gb, 2),
                generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    if args.formal:
        rg = pcfg['ranges']
        meta.update(purpose='formal', partition_train=rg['train'], partition_val=rg['val'],
                    partition_eval=rg['eval'], partition_test_SEALED=rg['sealed'],
                    protocol='Primary pool for a new p value, same layout as the p=0.004 '
                             'FORMAL pools (configs/partitions.yaml). The sealed block is '
                             'not read by any training or model-selection code.')
    with open(fn.replace('.npz', '.fingerprint.json'), 'w') as jf:
        json.dump(meta, jf, indent=2)

    print(f"[ext] wrote {fn}  ({size_gb:.1f} GB)")
    print(f"[ext] flips sha256 {flips_sha[:16]}  base_rate {meta['base_rate']}")
    if args.formal:
        print(f"[ext] FORMAL layout {pcfg['ranges']}; sealed block not to be read")
    else:
        print(f"[ext] every shot [0, {args.n:,}) is training data; nothing is reserved here")


if __name__ == '__main__':
    main()
