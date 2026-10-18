# Created: 2026-10-18
# Last modified: 2026-10-18
# Lowest selection-block p_L from --min-epoch, later epoch on a tie.
# Not converged: slope of the ratio over the last 20 epochs negative with t < -2.
import argparse
import csv
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--csv', required=True)
ap.add_argument('--min-epoch', type=int, required=True)
ap.add_argument('--epochs', type=int, required=True)
ap.add_argument('--weights-dir', required=True)
ap.add_argument('--out-json', required=True)
args = ap.parse_args()

rows = {int(r['epoch']): r for r in csv.DictReader(open(args.csv)) if r['model'] == args.model}
want = set(range(args.min_epoch, args.epochs + 1))
missing = sorted(want - set(rows))
if missing:
    raise SystemExit(f'{args.model}: no selection score for epochs {missing}')
cand = [rows[e] for e in sorted(want)]
best = min(cand, key=lambda r: (int(r['errors']), -int(r['epoch'])))
epoch = int(best['epoch'])
path = os.path.join(args.weights_dir, f'epoch{epoch:03d}.weights.h5')
if os.path.abspath(best['weights']) != os.path.abspath(path):
    raise SystemExit(f'{args.model}: epoch {epoch} was scored from {best["weights"]}, expected {path}')

slope = t = None
not_converged = None
tail = [rows[e] for e in range(args.epochs - 19, args.epochs + 1) if e in rows]
if len(tail) == 20:
    x = np.array([int(r['epoch']) for r in tail], float)
    y = np.array([float(r['ratio']) for r in tail])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
    slope = float(coef[0])
    resid = y - A @ coef
    se = float(np.sqrt(resid @ resid / (len(x) - 2) / ((x - x.mean()) @ (x - x.mean()))))
    t = slope / se if se > 0 else float('-inf') if slope < 0 else 0.0
    not_converged = bool(slope < 0 and t < -2)

out = dict(model=args.model, epoch=epoch, weights=path, sel_errors=int(best['errors']),
           sel_n=int(best['n']), sel_p_L=float(best['p_L']), sel_mwpm_errors=int(best['mwpm_errors']),
           sel_ratio=float(best['ratio']), epochs_scored=len(cand),
           last20_slope_per_epoch=slope, last20_slope_t=t, not_converged=not_converged)
json.dump(out, open(args.out_json, 'w'), indent=1)
print(json.dumps(out), flush=True)
