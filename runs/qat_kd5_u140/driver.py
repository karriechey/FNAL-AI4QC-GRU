# Created: 2026-10-28 | Last updated: 2026-10-28
# Runs QAT jobs from a job file (lines: tag seed [lr]), up to --max-jobs at once, each in its own
# GPU container. A run whose result says failed is retried once at half the LR (tag + '_halflr').
# New lines appended to the job file are picked up while the driver runs.
#   setsid nohup python3 driver.py --jobs jobs.txt --max-jobs 6 > logs/driver.log 2>&1 &
import argparse
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--jobs', default=os.path.join(HERE, 'jobs.txt'))
ap.add_argument('--max-jobs', type=int, default=6)
ap.add_argument('--lr', type=float, default=3e-4)
args = ap.parse_args()


def say(msg):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), msg, flush=True)


def read_jobs():
    out = []
    for line in open(args.jobs):
        f = line.split('#')[0].split()
        if f:
            out.append((f[0], int(f[1]), float(f[2]) if len(f) > 2 else args.lr))
    return out


def run_name(spec_tag, seed, lr):
    return f'{spec_tag}_seed{seed}' + ('_halflr' if lr < args.lr else '')


running, done, idle_logged = {}, set(), False
while True:
    queue = [j for j in read_jobs() if run_name(*j) not in done and run_name(*j) not in running]
    for name, (p, job) in list(running.items()):
        if p.poll() is None:
            continue
        del running[name]
        done.add(name)
        res = os.path.join(HERE, 'qat_runs', name, 'result.json')
        status = json.load(open(res))['status'] if os.path.exists(res) else f'no result (exit {p.returncode})'
        say(f'END {name}: {status}')
        tag, seed, lr = job
        if (status.startswith('failed') or status.startswith('no result')) and lr >= args.lr:
            with open(args.jobs, 'a') as f:
                f.write(f'{tag} {seed} {lr / 2}   # retry of {name}\n')
            say(f'queued retry of {name} at lr {lr / 2}')
    if queue:
        idle_logged = False
    while queue and len(running) < args.max_jobs:
        tag, seed, lr = queue.pop(0)
        name = run_name(tag, seed, lr)
        out = os.path.join(HERE, 'qat_runs', name)
        os.makedirs(out, exist_ok=True)
        cmd = ['bash', os.path.join(HERE, 'pod.sh'), f'qat_{name}', 'python3', 'train_qat.py', '--tag', name,
               '--spec-json', f'specs/{tag}.json', '--seed', str(seed), '--lr', str(lr), '--out', out]
        p = subprocess.Popen(cmd, stdout=open(os.path.join(out, 'run.log'), 'w'), stderr=subprocess.STDOUT, cwd=HERE)
        running[name] = (p, (tag, seed, lr))
        say(f'START {name} pid {p.pid}')
    if not running and not queue:
        if not idle_logged:
            say('queue empty')
        idle_logged = True
        if os.path.exists(os.path.join(HERE, 'STOP_DRIVER')):
            break
    time.sleep(30)
