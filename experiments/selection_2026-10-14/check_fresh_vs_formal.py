# Created: 2026-10-14
# Last modified: 2026-10-14
# Do fresh training shots repeat the shots of FORMAL [0, 10M)? Hashes every shot (detection
# events + logical flip) and counts how many shots of one fresh chunk also occur in the block.
# The same count for 100k pool shots from outside the block, [10M, 10.1M), is the chance level
# (frequent syndromes, e.g. no detection events, occur in any sample).
#   python check_fresh_vs_formal.py --pool <FORMAL.npz> --d 9 --p 0.003 --seed-base 1000000000000
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src'))
from circuit_generators import build_circuit
import fresh_shots as FS

ap = argparse.ArgumentParser()
ap.add_argument('--pool', required=True)
ap.add_argument('--d', type=int, required=True)
ap.add_argument('--p', type=float, required=True)
ap.add_argument('--seed-base', type=int, default=FS.SEED_BASE)
ap.add_argument('--n', type=int, default=100_000)
args = ap.parse_args()

rng = np.random.default_rng(0)
weights = None


def hashes(det, flip):
    global weights
    if weights is None:
        weights = rng.integers(1, 2**63, size=det.shape[1] + 1, dtype=np.uint64)
    out = np.empty(len(det), dtype=np.uint64)
    for i in range(0, len(det), 500_000):
        x = np.concatenate([det[i:i + 500_000], flip[i:i + 500_000, None]], axis=1).astype(np.uint64)
        out[i:i + 500_000] = x @ weights
    return out


z = np.load(args.pool)
det, flips = z['det_evts'], z['flips'].reshape(-1).astype(np.uint8)
block = hashes(det[:10_000_000], flips[:10_000_000])
control = hashes(det[10_000_000:10_000_000 + args.n], flips[10_000_000:10_000_000 + args.n])

circ = build_circuit(args.d, args.p, args.d)
seed = FS.chunk_seed(0, 0, args.seed_base)
fdet, fobs = circ.compile_detector_sampler(seed=seed).sample(args.n, separate_observables=True)
fresh = hashes(fdet.astype(np.uint8), fobs[:, 0].astype(np.uint8))

block_sorted = np.sort(block)
res = dict(pool=args.pool, d=args.d, p=args.p, fresh_seed=seed, n=args.n,
           det_rate_pool=float(det[:args.n].mean()), det_rate_fresh=float(fdet.mean()),
           fresh_in_block=int(np.isin(fresh, block_sorted).sum()),
           control_in_block=int(np.isin(control, block_sorted).sum()),
           fresh_equal_to_pool_row_0_on=int((fresh == block[:args.n]).sum()))
print(json.dumps(res))
