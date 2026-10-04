#!/usr/bin/env python3
# Created: 2026-10-04
# Last modified: 2026-10-04
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
import sys

import driver as D

ap = argparse.ArgumentParser()
ap.add_argument('name')
ap.add_argument('--baseline-history', required=True)
a = ap.parse_args()
job = {j['name']: j for j in D.read_jobs()}[a.name]
if D.status(a.name):
    sys.exit(f'{a.name} already has a run folder ({D.status(a.name)}); not starting it again')
if any(v == 'STAB' for v in (job['peak_lr'], job['warmup'], job['clip'])) or job['cond'] != '-':
    sys.exit(f'{a.name} depends on the promotion or a condition; run it through the driver')
if D.host_avail_gb() < int(job['need_host_gb']):
    sys.exit(f"host available {D.host_avail_gb()} GB < {job['need_host_gb']} GB")
hb = json.load(open(a.baseline_history))
threshold = D.KILL_FACTOR * hb['val_ler'][D.KILL_EPOCH - 1]
D.log(f"NOTE  run_one: starting {a.name} alongside the queue's running job (amendment 5); "
      f"epoch-{D.KILL_EPOCH} kill threshold {threshold:.6f}")
D.run_job(job, threshold, False)
