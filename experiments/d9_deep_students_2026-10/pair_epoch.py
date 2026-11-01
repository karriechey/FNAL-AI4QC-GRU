#!/usr/bin/env python3
# Created: 2026-10-24
# Last modified: 2026-10-25
# One status.log line pairing a run's validation p_L at --epoch with the reference runs at the
# same epoch: kd9_3x140 from the kd pilot history.json, the tapers from their own run folders.
# Called by run_student.sh every 10 epochs.
#
# p_L is the integer validation error count over the block, never the four-decimal val_ler the
# Keras epoch line prints: at 0.0025 that rounding moves a ratio by up to 2%, near the 3%
# single-seed margin. Epochs are keyed by the number train_student.py prints, so a run that was
# paused and resumed (run_part*.log plus run.log, and run_part1.log is a copy of run.log until
# the resume) gives one entry per epoch and is checked for being exactly 1...N.
import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

MWPM = 0.002267   # d9 p=0.004 FORMAL pool, validation block
N_VAL = 200000    # validation block FORMAL [15.0M, 15.2M)
HERE = os.path.dirname(os.path.abspath(__file__))
VAL_LINE = re.compile(r'^\[gru\] epoch (\d+): val_loss=\S+ val_ler=\S+ \(([\d,]+)/([\d,]+) errors\)')


def epochs_from_run_dir(out):
    by_epoch = {}
    for log in sorted(glob.glob(os.path.join(out, 'seed0', 'run_part*.log'))) + \
               [os.path.join(out, 'seed0', 'run.log')]:
        if not os.path.exists(log):
            continue
        for line in open(log, errors='replace'):
            m = VAL_LINE.match(line)
            if not m:
                continue
            e, errs, n = int(m.group(1)), int(m.group(2).replace(',', '')), int(m.group(3).replace(',', ''))
            if n != N_VAL:
                sys.exit(f'{out}: epoch {e} validated on {n} shots, expected {N_VAL}')
            p = errs / n
            if by_epoch.get(e, p) != p:
                sys.exit(f'{out}: epoch {e} appears twice with different p_L '
                         f'({by_epoch[e]:.6f}, {p:.6f})')
            by_epoch[e] = p
    if by_epoch and sorted(by_epoch) != list(range(1, max(by_epoch) + 1)):
        missing = sorted(set(range(1, max(by_epoch) + 1)) - set(by_epoch))
        sys.exit(f'{out}: epochs are not 1...{max(by_epoch)}, missing {missing[:10]}')
    return by_epoch


def kd_pilot_epochs(run):
    for root in (os.path.expanduser('~/FNAL-AI4QC-GRU-kd/experiments/kd_pilot_p004_2026-10/runs'),
                 os.path.join(HERE, '..', 'kd_pilot_p004_2026-10', 'runs')):
        for h in sorted(glob.glob(os.path.join(root, f'{run}_2026*', 'seed0', 'runs', '*.history.json'))):
            d = json.load(open(h))
            if d.get('val_errors'):
                v = [float(x) / N_VAL for x in d['val_errors']]
            elif d.get('val_ler'):   # an older history without the counts
                v = [round(float(x) * N_VAL) / N_VAL for x in d['val_ler']]
            else:
                continue
            return {i + 1: p for i, p in enumerate(v)}
    return {}


def local_epochs(run):
    outs = sorted(glob.glob(os.path.join(HERE, 'runs', f'{run}_2026*')))
    return epochs_from_run_dir(outs[-1]) if outs else {}


ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--epoch', type=int, required=True)
ap.add_argument('--quiet-missing', action='store_true')
args = ap.parse_args()

mine = epochs_from_run_dir(args.out)
if args.epoch not in mine:
    sys.exit(f'{args.run}: no epoch {args.epoch} in the log ({len(mine)} epochs)')
v = mine[args.epoch]

parts = []
for ref, got in (('kd9_3x140', kd_pilot_epochs('kd9_3x140')),
                 ('kd9_taper4', local_epochs('kd9_taper4')),
                 ('kd9_taper3_t100', local_epochs('kd9_taper3_t100'))):
    if ref == args.run:
        continue
    if args.epoch in got:
        r = got[args.epoch]
        parts.append(f'{ref} {r:.6f} ({r / MWPM:.2f}x)')
    elif not args.quiet_missing:
        parts.append(f'{ref} no epoch {args.epoch}')

stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
print(f'{stamp} {args.run} paired epoch {args.epoch}: '
      f'val p_L {v:.6f} ({v / MWPM:.2f}x), ' + ', '.join(parts))
