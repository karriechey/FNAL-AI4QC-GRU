# Created: 2026-10-14
# Last modified: 2026-10-16

"""Keep a copy of every epoch's .lastepoch.weights.h5 for runs already training.

train_student.py overwrites that file each epoch. The epoch is read from the optimizer's update
count in the file. Copies go to <run folder>/seed<S>/epochs/epoch<NNN>.weights.h5; exits when
no run's container is up.

~/tfenv/bin/python snapshot_epochs.py <run folder> [<run folder> ...]
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import time

import h5py

FIRST = 40
MIN_FREE_GB = 1                       # /home is shared; skip copies below this


def steps_per_epoch(run):
    env = open(os.path.join(run, 'env.txt')).read()
    shots = int(re.search(r'^FRESH_SHOTS=(\d+)', env, re.M).group(1))
    return shots // int(re.search(r'^BATCH=(\d+)', env, re.M).group(1))


def updates(path):
    with h5py.File(path, 'r') as f:
        return int(f['optimizer/vars/0'][()])


def container_up(run):
    arm = os.path.basename(run).split('_u')[0]
    names = subprocess.run(['podman', 'ps', '--format', '{{.Names}}'],
                           capture_output=True, text=True).stdout.split()
    return f'probe_{arm}' in names


runs = [os.path.abspath(r) for r in sys.argv[1:]]
steps = {r: steps_per_epoch(r) for r in runs}
seen = {}
was_up = True
while True:
    up = [r for r in runs if container_up(r)]
    for run in runs:
        for src in glob.glob(os.path.join(run, 'seed*', 'ckpt', '*.lastepoch.weights.h5')):
            mtime = os.path.getmtime(src)
            if seen.get(src) == mtime or time.time() - mtime < 5:
                continue
            try:
                n = updates(src)
            except OSError:                  # caught mid-write; next pass
                continue
            if n % steps[run]:
                continue
            epoch = n // steps[run]
            out = os.path.join(os.path.dirname(os.path.dirname(src)), 'epochs')
            dst = os.path.join(out, f'epoch{epoch:03d}.weights.h5')
            if epoch >= FIRST and not os.path.exists(dst):
                if shutil.disk_usage(run).free < MIN_FREE_GB * 2**30:
                    print(time.strftime('%FT%TZ', time.gmtime()), 'low disk, skipped', dst, flush=True)
                    seen[src] = mtime
                    continue
                os.makedirs(out, exist_ok=True)
                shutil.copy2(src, dst + '.tmp')
                if updates(dst + '.tmp') != n or os.path.getmtime(src) != mtime:
                    os.remove(dst + '.tmp')  # file changed while copying
                    continue
                os.rename(dst + '.tmp', dst)
                print(time.strftime('%FT%TZ', time.gmtime()), 'saved', dst, flush=True)
            seen[src] = mtime
    if not up and not was_up:         # one more pass after the last container stops
        break
    was_up = bool(up)
    time.sleep(10)
