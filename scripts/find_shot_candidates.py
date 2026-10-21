#!/usr/bin/env python3
# Created: 2026-10-12
# Last updated: 2026-10-12

"""Find evaluation-block shots where MWPM is wrong and a GRU checkpoint is right (animateQEC RealShot).

Prints a table; writes a file only with --out.

~/tfenv/bin/python find_shot_candidates.py --src ~/FNAL-AI4QC-GRU-fresh2/src \
    --weights <ckpt>/...best.weights.h5 --units 280 280 280
"""
import argparse
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="src/ of the code copy that trained the checkpoint")
ap.add_argument("--weights", required=True, help=".best.weights.h5")
ap.add_argument("--units", type=int, nargs="+", required=True)
ap.add_argument("--d", type=int, default=9)
ap.add_argument("--p", type=float, default=0.004)
ap.add_argument("--max-events", type=int, default=27)
ap.add_argument("--n-backup", type=int, default=200, help="fewest-event shots scored as backups")
ap.add_argument("--show", type=int, default=15)
ap.add_argument("--out", help="optional .npz of the candidate rows (stays on grace1)")
args = ap.parse_args()

sys.path.insert(0, os.path.abspath(os.path.expanduser(args.src)))
import partitions
import pool_rows
from circuit_generators import build_circuit
from StudentModels import build_gru_student, detector_sequence_layout, to_sequence

d, r, p = args.d, args.d, args.p
pool = partitions.pool_path(d, "formal")
a, b = partitions.block("eval")
s0, s1 = partitions.block("sealed")
assert b <= s0 or a >= s1, "evaluation block overlaps the sealed block"
print(f"pool {pool}\nevaluation block [{a}, {b}) ({b - a} shots); sealed [{s0}, {s1}) not read")

det = pool_rows.rows(pool, "det_evts", a, b).astype(np.uint8)
flips = pool_rows.rows(pool, "flips", a, b).reshape(-1).astype(bool)
n_ev = det.sum(1)
few = np.flatnonzero((n_ev >= 4) & (n_ev <= args.max_events))
print(f"shots with 4..{args.max_events} events: {len(few)}")

import pymatching
circ = build_circuit(d, p, r)
m = pymatching.Matching.from_detector_error_model(circ.detector_error_model(decompose_errors=True))
mwpm = m.decode_batch(det[few])[:, 0].astype(bool)
fail = few[mwpm != flips[few]]
backup = few[np.argsort(n_ev[few], kind="stable")[:args.n_backup]]
print(f"MWPM failures among them: {len(fail)}")

model = build_gru_student(d=d, rounds=r, inputs="evts", units=tuple(args.units), hidden=(), p=p,
                          reset_after=False, unroll=False)
model.load_weights(args.weights)
score = np.unique(np.concatenate([fail, backup]))
logit = model.predict(to_sequence(det[score], d, r, p), batch_size=4096, verbose=0).reshape(-1)
p_flip = dict(zip(score, 1 / (1 + np.exp(-logit))))
mwpm_pred = dict(zip(few, mwpm))

_, _, index_map = detector_sequence_layout(d, r, p)
t_of = np.full(det.shape[1], -1)
for t, row in enumerate(index_map):
    t_of[row[row >= 0]] = t


def show(rows, title):
    print(f"\n{title}")
    print(f"{'pool shot':>10} {'block i':>8} {'events':>6}  {'per round t=0..' + str(r):<32} "
          f"{'truth':>5} {'MWPM':>5} {'GRU P(flip)':>11} {'GRU':>4}")
    for i in rows[:args.show]:
        per = np.bincount(t_of[np.flatnonzero(det[i])], minlength=r + 1)
        g = p_flip[i] > 0.5
        print(f"{a + i:>10} {i:>8} {n_ev[i]:>6}  {' '.join(map(str, per)):<32} "
              f"{int(flips[i]):>5} {int(mwpm_pred[i]):>5} {p_flip[i]:>11.4f} {int(g):>4}")


gru_right = np.array([(p_flip[i] > 0.5) == flips[i] for i in fail], dtype=bool)
main = fail[gru_right]
main = main[np.argsort(n_ev[main], kind="stable")]
print(f"\nMWPM wrong, GRU right: {len(main)} of {len(fail)} MWPM failures "
      f"(GRU also wrong on {int((~gru_right).sum())})")
show(main, "candidates: MWPM wrong, GRU right (fewest events first)")
both = [i for i in backup if mwpm_pred[i] == flips[i] and (p_flip[i] > 0.5) == flips[i]]
show(both, "backups: both right")

if args.out:
    keep = np.concatenate([main[:args.show], np.array(both[:args.show], dtype=int)])
    np.savez(args.out, block_index=keep, pool_index=a + keep, n_events=n_ev[keep],
             truth=flips[keep], p_flip=np.array([p_flip[i] for i in keep]),
             mwpm=np.array([mwpm_pred[i] for i in keep]), weights=args.weights)
    print(f"\nwrote {args.out}")
