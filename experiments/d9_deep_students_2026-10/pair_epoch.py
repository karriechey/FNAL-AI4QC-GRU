#!/usr/bin/env python3
# Created: 2026-10-24
# Last modified: 2026-10-25
# One status.log line pairing a run's validation p_L at --epoch with the reference runs at the
# same epoch: kd9_3x140 from the kd pilot history.json, kd9_taper4 from its own run folder here.
# Called by run_student.sh every 10 epochs.
import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

MWPM = 0.002267   # d9 p=0.004 FORMAL pool, validation block
HERE = os.path.dirname(os.path.abspath(__file__))
EPOCH_LINE = re.compile(r'^\d+/\d+ - \d+s - .*val_ler: ([0-9.eE+-]+)')


# a resumed run keeps the epochs before the pause in run_part*.log
def val_lers_from_run_dir(out):
    v = []
    for log in sorted(glob.glob(os.path.join(out, 'seed0', 'run_part*.log'))) + \
               [os.path.join(out, 'seed0', 'run.log')]:
        if os.path.exists(log):
            v += [float(m.group(1)) for m in map(EPOCH_LINE.match, open(log, errors='replace')) if m]
    return v


def kd_pilot_val_lers(run):
    for root in (os.path.expanduser('~/FNAL-AI4QC-GRU-kd/experiments/kd_pilot_p004_2026-10/runs'),
                 os.path.join(HERE, '..', 'kd_pilot_p004_2026-10', 'runs')):
        for h in sorted(glob.glob(os.path.join(root, f'{run}_2026*', 'seed0', 'runs', '*.history.json'))):
            v = json.load(open(h)).get('val_ler', [])
            if v:
                return [float(x) for x in v]
    return []


def local_val_lers(run):
    outs = sorted(glob.glob(os.path.join(HERE, 'runs', f'{run}_2026*')))
    return val_lers_from_run_dir(outs[-1]) if outs else []


ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--epoch', type=int, required=True)
args = ap.parse_args()

mine = val_lers_from_run_dir(args.out)
if len(mine) < args.epoch:
    sys.exit(f'{args.run}: run.log has {len(mine)} epochs, asked for {args.epoch}')
v = mine[args.epoch - 1]

parts = []
for ref, lers in (('kd9_3x140', kd_pilot_val_lers('kd9_3x140')),
                  ('kd9_taper4', local_val_lers('kd9_taper4'))):
    if ref == args.run:
        continue
    if len(lers) >= args.epoch:
        r = lers[args.epoch - 1]
        parts.append(f'{ref} {r:.6f} ({r / MWPM:.2f}x)')
    else:
        parts.append(f'{ref} no epoch {args.epoch}')

stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
print(f'{stamp} {args.run} paired epoch {args.epoch}: '
      f'val p_L {v:.6f} ({v / MWPM:.2f}x), ' + ', '.join(parts))
