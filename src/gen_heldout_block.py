#!/usr/bin/env python3
# Created: 2026-10-06
# Last modified: 2026-10-06
"""Generate a held-out block: a separate pool file used only as a final test set.

Same circuit and noise as the FORMAL pools (generate_pools.build_circuit: rotated memory-Z,
rounds = d, all four Stim noise channels = p) and the same sampler (generate_pools.sample_pool,
chunked), from a generation seed never used for any pool. The seed is checked against every
seed in configs/partitions.yaml (FORMAL, TRAINONLY, held-out, and the seeds listed as used
elsewhere) and against the .fingerprint.json files in the output folder; any match stops.

Nothing is scored here: no MWPM decode, no model. The whole file is a held-out block, so
partitions.sealed_block treats every shot of it as sealed (eval_student_on_tail.py refuses
it without --score-sealed, and logs any such scoring).

Writes  <out-dir>/data_d{d}_p{p:.3f}_r{d}_HELDOUT_seed{seed}.npz   measurements, det_evts, flips
        <out-dir>/data_d{d}_p{p:.3f}_r{d}_HELDOUT_seed{seed}.fingerprint.json
and prints the lines to add to configs/partitions.yaml.

    python gen_heldout_block.py --d 5 --p 0.004 --n 2000000 --seed 7001 \\
        --out-dir ~/pools_d5_p004_19M
"""
import argparse
import datetime
import glob
import hashlib
import json
import os

import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, required=True)
    ap.add_argument('--p', type=float, required=True)
    ap.add_argument('--n', type=int, required=True, help='number of shots')
    ap.add_argument('--seed', type=int, required=True, help='Stim generation seed, never used before')
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()

    import partitions as P
    used = P.used_seeds()
    out_dir = os.path.expanduser(a.out_dir)
    for f in glob.glob(os.path.join(out_dir, '*.fingerprint.json')):
        s = json.load(open(f)).get('gen_seed')
        if s is not None:
            used.add(int(s))
    if a.seed in used:
        raise SystemExit(f'[heldout] seed {a.seed} is already used (seeds in use: {sorted(used)}). STOP.')

    name = f'data_d{a.d}_p{a.p:.3f}_r{a.d}_HELDOUT_seed{a.seed}'
    fn = os.path.join(out_dir, name + '.npz')
    if os.path.exists(fn):
        raise SystemExit(f'[heldout] {fn} exists; not overwritten. STOP.')

    import stim
    from generate_pools import build_circuit, sample_pool
    t0 = datetime.datetime.now(datetime.timezone.utc)
    circ = build_circuit(a.d, a.p)
    measurements, det_evts, flips = sample_pool(circ, a.n, a.seed)
    np.savez(fn, measurements=measurements, det_evts=det_evts, flips=flips)
    meta = dict(
        path=fn, d=a.d, p=a.p, rounds=a.d, n_total=a.n, gen_seed=a.seed, purpose='held_out',
        partition_held_out=[0, a.n],
        protocol=('Held-out test block generated 2026-10-06 with a seed never used for any '
                  'pool; same circuit and noise as the FORMAL pools. Every shot is sealed: '
                  'not read by training or model selection; scoring it needs '
                  'eval_student_on_tail.py --score-sealed, which logs the access.'),
        scored=False, format='full: measurements, det_evts, flips',
        measurements_shape=str(measurements.shape), det_evts_shape=str(det_evts.shape),
        flips_sha256=hashlib.sha256(np.ascontiguousarray(flips).tobytes()).hexdigest(),
        file_sha256=sha256_file(fn), stim_version=stim.__version__,
        numpy_version=np.__version__,
        generated_utc=t0.strftime('%Y-%m-%dT%H:%M:%SZ'),
        size_gb=round(os.path.getsize(fn) / 1024 ** 3, 3))
    json.dump(meta, open(fn.replace('.npz', '.fingerprint.json'), 'w'), indent=2)
    print(f"[heldout] wrote {fn}: {a.n:,} shots, seed {a.seed}, flips_sha256 "
          f"{meta['flips_sha256'][:16]}..., file_sha256 {meta['file_sha256'][:16]}... (not scored)")
    print(json.dumps({k: meta[k] for k in ('d', 'p', 'rounds', 'n_total', 'gen_seed',
                                             'flips_sha256', 'file_sha256', 'generated_utc')}))


if __name__ == '__main__':
    main()
