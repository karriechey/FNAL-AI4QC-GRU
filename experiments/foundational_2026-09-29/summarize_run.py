# Created: 2026-09-29
# Last modified: 2026-09-29 (GPU peak from the run's own process: first new GPU process after the run's start; resumed runs report the resumed segment's runtime)
"""One-run summary for the foundational queue, printed as JSON.

    python summarize_run.py <run folder> <resources.tsv> <container name>

Reads <run>/seed0: eval_best_ckpt.csv (GRU and MWPM logical error rates on the evaluation
block), ckpt/*.best.json (chosen epoch), runs/*.history.json (losses per epoch), run.log
(parameter count, start/end time). Peak host memory of the container and peak GPU memory
of its process come from the driver's 30-second samples in resources.tsv, so they are
lower bounds on the true peaks.
"""
import csv
import datetime as dt
import glob
import json
import os
import re
import sys

import numpy as np

run, res, cname = sys.argv[1:4]
sd = os.path.join(run, 'seed0')
ev = list(csv.DictReader(open(os.path.join(sd, 'eval_best_ckpt.csv'))))[-1]
best = json.load(open(glob.glob(os.path.join(sd, 'ckpt', '*.best.json'))[0]))
hist = json.load(open(sorted(glob.glob(os.path.join(sd, 'runs', '*.history.json')))[-1]))
log = open(os.path.join(sd, 'run.log')).read()
b = int(best['best_epoch']) - 1                       # 0-indexed row of the chosen epoch
t = re.findall(r'^(\S+Z) (start|end)$', log, re.M)
times = {k: dt.datetime.strptime(v, '%Y-%m-%dT%H:%M:%SZ') for v, k in t}

# Peaks from the driver's samples. Host: the container's memory. GPU: the run's own
# process, identified as the first GPU process that appears after the run's start and was
# not present before it (jobs start at least 60 s apart, samples are every 30 s).
start, end = t[0][0], t[-1][0]
host = 0.0
unit = {'kB': 1e-6, 'MB': 1e-3, 'GB': 1.0}
first_seen, gpu_by_pid = {}, {}
for line in open(res):
    f = line.rstrip('\n').split('\t')
    if f[1] == 'gpu_pid_mib':
        pid, mib = [s_.strip() for s_ in f[2].split(',')]
        first_seen.setdefault(pid, f[0])
        if start <= f[0] <= end:
            gpu_by_pid[pid] = max(gpu_by_pid.get(pid, 0), int(mib))
    if not (start <= f[0] <= end):
        continue
    if f[1] == 'container_mem' and f[2] == cname:
        m = re.match(r'([\d.]+)([kMG]B)', f[3].split('/')[0].strip())
        host = max(host, float(m.group(1)) * unit[m.group(2)])
new_pids = sorted((ts, pid) for pid, ts in first_seen.items() if ts >= start and pid in gpu_by_pid)
gpu_pid = new_pids[0][1] if new_pids else None
gpu = gpu_by_pid.get(gpu_pid, 0) if gpu_pid else 0

out = dict(
    run=os.path.basename(run),
    params=int(re.search(r'params=([\d,]+)', log).group(1).replace(',', '')),
    p_L=float(ev['p_L']), mwpm_p_L=float(ev['mwpm_p_L']), ratio=round(float(ev['ratio']), 4),
    best_epoch=b + 1, epochs=len(hist['loss']),
    best_val_loss=round(float(hist['val_loss'][b]), 5),
    best_val_ler=round(float(hist['val_ler'][b]), 6),
    best_train_loss=round(float(hist['loss'][b]), 5),
    final_train_loss=round(float(hist['loss'][-1]), 5),
    final_val_loss=round(float(hist['val_loss'][-1]), 5),
    min_val_loss_epoch=int(np.argmin(hist['val_loss'])) + 1,
    start=start, end=end,
    runtime_h=round((times['end'] - times['start']).total_seconds() / 3600, 2),
    peak_host_gb=round(host, 1), peak_gpu_mib=gpu, gpu_pid=gpu_pid)
print(json.dumps(out, indent=1))
