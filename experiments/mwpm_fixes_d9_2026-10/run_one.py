#!/usr/bin/env python3
# Created: 2026-10-04
# Last modified: 2026-10-05 (--resume: continue a run paused by pause_run.sh)
"""Run one row of jobs.tsv now, alongside the job the queue driver is running (amendment 5).

Uses driver.run_job, so the run gets exactly the queue's launcher settings, kill rules (NaN,
5 epochs above 1.5x its own best, epoch-60 above 2x R02), driver.log lines and results.tsv
row. The GPU check of the driver is not applied (the other process on the GPU is the
queue's own job); host memory must have the row's need_host_gb available. The queue driver
sees this row's run folder and does not start it again.

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
    # A resumed run keeps the settings it started with: the promotion in force at its launch,
    # read from its own command.txt, so a later re-promotion cannot change a run mid-way.
    env = dict(kv.split('=', 1) for kv in open(os.path.join(src, 'command.txt')).read().split()
               if '=' in kv and kv.split('=', 1)[0] in ('LR', 'LR_WARMUP_EPOCHS', 'GLOBAL_CLIPNORM'))
    job, _ = D.resolve(job, dict(peak_lr=env['LR'], warmup=env['LR_WARMUP_EPOCHS'],
                                 clip=env['GLOBAL_CLIPNORM']), None)
elif job['cond'] != '-':
    sys.exit(f'{a.name} depends on a condition; run it through the driver')
if D.host_avail_gb() < int(job['need_host_gb']):
    sys.exit(f"host available {D.host_avail_gb()} GB < {job['need_host_gb']} GB")
hb = json.load(open(a.baseline_history))
threshold = D.KILL_FACTOR * hb['val_ler'][D.KILL_EPOCH - 1]
D.log(f"NOTE  run_one: {'resuming' if a.resume else 'starting'} {a.name} alongside the queue's "
      f"running job (amendment 5); epoch-{D.KILL_EPOCH} kill threshold {threshold:.6f}")
D.run_job(job, threshold, False, resume_from=(src if a.resume else None))
