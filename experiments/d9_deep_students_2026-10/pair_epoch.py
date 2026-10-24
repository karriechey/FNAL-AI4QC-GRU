#!/usr/bin/env python3
# Created: 2026-10-24
# Last modified: 2026-10-24
# One status.log line pairing a run's validation p_L at --epoch with kd9_3x140 at the same
# epoch, read from the kd pilot history.json. Called by run_student.sh every 10 epochs.
import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

MWPM = 0.002267   # d9 p=0.004 FORMAL pool, validation block
REF = 'kd9_3x140'
EPOCH_LINE = re.compile(r'^\d+/\d+ - \d+s - .*val_ler: ([0-9.eE+-]+)')


def ref_val_ler(epoch):
    for root in (os.path.expanduser('~/FNAL-AI4QC-GRU-kd/experiments/kd_pilot_p004_2026-10/runs'),
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                              'kd_pilot_p004_2026-10', 'runs')):
        for h in sorted(glob.glob(os.path.join(root, f'{REF}_2026*', 'seed0', 'runs', '*.history.json'))):
            v = json.load(open(h)).get('val_ler', [])
            if len(v) >= epoch:
                return float(v[epoch - 1])
    return None


ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--epoch', type=int, required=True)
args = ap.parse_args()

log = os.path.join(args.out, 'seed0', 'run.log')
mine = [float(m.group(1)) for m in map(EPOCH_LINE.match, open(log, errors='replace')) if m]
if len(mine) < args.epoch:
    sys.exit(f'{args.run}: run.log has {len(mine)} epochs, asked for {args.epoch}')
v = mine[args.epoch - 1]
r = ref_val_ler(args.epoch)
stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
ref = f'{REF} {r:.6f} ({r / MWPM:.2f}x)' if r is not None else f'{REF} no epoch {args.epoch}'
print(f'{stamp} {args.run} paired epoch {args.epoch}: val p_L {v:.6f} ({v / MWPM:.2f}x), {ref}')
