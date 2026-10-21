#!/usr/bin/env python3
# Created: 2026-10-04
# Last modified: 2026-10-06 (--need-host-gb).
# Earlier 2026-10-05 (--resume: continue a run paused by pause_run.sh)

"""Run one row of jobs.tsv now, next to the job the queue driver is running.

Uses driver.run_job, so the run gets the queue's launcher settings, kill rules, log lines and
results.tsv row. Only host memory is checked. The driver sees the run folder and skips the row.

setsid nohup ~/tfenv/bin/python run_one.py <name> --baseline-history <R02 history.json> &
"""
import argparse
import json
import os
import sys

import driver as D

ap = argparse.ArgumentParser()
ap.add_argument('name')
ap.add_argument('--baseline-history', required=True)
ap.add_argument('--need-host-gb', type=int, default=None,
                help="host memory required to start, replacing the row's need_host_gb (set from the "
                     "run's measured footprint plus a margin)")
ap.add_argument('--resume', action='store_true',
                help='continue the paused run of this row (its latest folder has PAUSED) in a '
                     'new folder; the paused folder is not changed')
a = ap.parse_args()
job = {j['name']: j for j in D.read_jobs()}[a.name]
src = D.run_dir(a.name)
if a.resume:
    if not (src and os.path.exists(os.path.join(src, 'PAUSED'))):
        sys.exit(f'{a.name}: no paused run folder to resume')
    if os.popen(f"podman ps -q --filter name=^md9_{a.name}$").read().strip():
        sys.exit(f'{a.name}: its container is still running')
elif D.status(a.name):
    sys.exit(f'{a.name} already has a run folder ({D.status(a.name)}); not starting it again')
if any(v == 'STAB' for v in (job['peak_lr'], job['warmup'], job['clip'])):
    if not a.resume:
        sys.exit(f'{a.name} depends on the promotion; run it through the driver')
    # a resumed run keeps the promotion in force at its launch (read from its command.txt)
    env = dict(kv.split('=', 1) for kv in open(os.path.join(src, 'command.txt')).read().split()
               if '=' in kv and kv.split('=', 1)[0] in ('LR', 'LR_WARMUP_EPOCHS', 'GLOBAL_CLIPNORM'))
    job, _ = D.resolve(job, dict(peak_lr=env['LR'], warmup=env['LR_WARMUP_EPOCHS'],
                                 clip=env['GLOBAL_CLIPNORM']), None)
elif job['cond'] != '-':
    sys.exit(f'{a.name} depends on a condition; run it through the driver')
need = a.need_host_gb or int(job['need_host_gb'])
if D.host_avail_gb() < need:
    sys.exit(f"host available {D.host_avail_gb()} GB < {need} GB")
hb = json.load(open(a.baseline_history))
threshold = D.KILL_FACTOR * hb['val_ler'][D.KILL_EPOCH - 1]
D.log(f"NOTE  run_one: {'resuming' if a.resume else 'starting'} {a.name} alongside the queue's "
      f"running job (amendment 5); epoch-{D.KILL_EPOCH} kill threshold {threshold:.6f}")
D.run_job(job, threshold, False, resume_from=(src if a.resume else None))
