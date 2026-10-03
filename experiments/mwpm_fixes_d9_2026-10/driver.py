#!/usr/bin/env python3
# Created: 2026-10-03
# Last modified: 2026-10-03
"""Queue driver for the d=9 MWPM-fix runs (prereg/mwpm_fixes_d9_2026-10.md).

One run on the GPU at a time, in the order of jobs.tsv, with the rules fixed in the
pre-registration:

  GPU       before each launch, no compute process of another user on the GPU; if there is
            one, wait and re-check every 15 minutes
  host      host memory available >= the job's need_host_gb
  budget    MAX_PARAMS = the single-layer u280 count at d=9 (303,521); the launcher and
            train_student.py refuse a model over it
  kill      training loss nan -> stopped (DIVERGED). At 25% of the run's epochs, validation
            p_L above 1.25 x the baseline run's validation p_L at 25% of its epochs (R02,
            epoch 20 of 80) -> stopped (KILLED). Both recorded with the epoch.
  win       p_L / MWPM on the evaluation block < 1 with McNemar p_exact < 0.01
  promote   after 9-1, 9-2, 9-3 finish, the one with the lowest p_L / MWPM sets the peak LR,
            warmup and clipnorm of every later row marked STAB (stability.json). If all
            three were stopped, the driver stops here.
  9-6       only if the promoted run's best epoch is in the last 10% of its epochs;
            otherwise skipped, with the reason logged
  9-8       needs the stacked GRU (two layers), which needs the evaluation script to rebuild
            two layers; skipped until that change is approved and present
  9-10      the best config so far (lowest ratio among 9-1..9-9), seed 1. If a run wins, 9-10
            runs next instead of last.

Every finished, stopped or skipped run gets one row in results.tsv with the columns of
GRU_all_runs_2026-10-01.csv. Runs from the launcher gh200_run_student_scaling.sh in the
qdec:tf215 container, from the checkout this file sits in. The sealed block is never read
(the launcher scores the evaluation block [15.2M, 17M) only).

    setsid nohup ~/tfenv/bin/python driver.py --baseline-history <R02 history.json> \\
        > /dev/null 2>&1 < /dev/null &
    python3 driver.py --dry-run --baseline-history <...>   # print the plan, launch nothing
"""
import argparse
import csv
import datetime
import glob
import json
import math
import os
import re
import subprocess
import sys
import time

DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(DIR, '..', '..'))
JOBS = os.path.join(DIR, 'jobs.tsv')
LOGF = os.path.join(DIR, 'driver.log')
RESULTS = os.path.join(DIR, 'results.tsv')
STAB = os.path.join(DIR, 'stability.json')
IMG = 'qdec:tf215'
PREFIX = 'md9_'
HOME = os.path.expanduser('~')
D = 9
MAX_PARAMS = 303521                      # single-layer u280 at d=9
POOL = f'{HOME}/pools_d9_p004_19M/data_d9_p0.004_r9_FORMAL.npz'
TRAINONLY = f'{HOME}/pools_d9_p004_19M/data_d9_p0.004_r9_TRAINONLY_seed{{}}.npz'
DATA = {   # training shots -> (extension seeds, counts); primary prefix 15M as in R02/R16
    '100M': ([48, 49], [35_000_000, 50_000_000]),
    '200M': ([48, 49, 58, 59], [35_000_000, 50_000_000, 50_000_000, 50_000_000]),
}
WARM_START = {   # name in jobs.tsv -> glob of the source model
    'R01': f'{HOME}/FNAL-AI4QC-GRU-compact/experiments/queue12_2026-09-30/'
           'R01_d7_u280_200M_2*/seed0/ckpt/*.best.keras',
}
GPU_WAIT_S = 900
POLL_S = 60
KILL_FRACTION = 0.25
KILL_FACTOR = 1.25
CSV_COLS = ['Done?', 'Width', 'Parameters', 'd', 'r', 'LR Schedule', 'LR', 'Data', 'Seed', 'p',
            'Batch', 'Max Epoch', 'Best Ckpt', 'MWPM', 'p_L', 'Relative to MWPM', 'Val_loss',
            'reset_after', 'Checkpoint criterion', 'runtime (h)', 'Date']


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


DRY = False


def log(msg):
    line = f'[mwpm-d9 {now()}] {msg}'
    print(line, flush=True)
    if not DRY:                       # a dry run writes no file
        with open(LOGF, 'a') as fh:
            fh.write(line + '\n')


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def read_jobs():
    """jobs.tsv rows as dicts, '#' lines skipped."""
    rows = []
    with open(JOBS) as fh:
        lines = [l for l in fh if l.strip() and not l.startswith('#')]
    for r in csv.DictReader(lines, delimiter='\t'):
        rows.append({k: v.strip() for k, v in r.items()})
    return rows


def run_dir(name):
    found = sorted(glob.glob(os.path.join(DIR, f'{name}_2*')))
    return found[-1] if found else None


def status(name):
    """'' (not started), running, done, killed, diverged, failed, skipped."""
    out = run_dir(name)
    if out is None:
        return ''
    for f, s in (('SKIPPED', 'skipped'), ('KILLED', 'killed'), ('DIVERGED', 'diverged')):
        if os.path.exists(os.path.join(out, f)):
            return s
    if glob.glob(os.path.join(out, 'seed*', 'COMPLETE')):
        return 'done'
    if os.path.exists(os.path.join(out, 'exit_code')):
        return 'failed'
    return 'running'


def eval_row(out):
    """The evaluation-block row of a finished run (eval_best_ckpt.csv), or None."""
    f = glob.glob(os.path.join(out, 'seed*', 'eval_best_ckpt.csv'))
    if not f:
        return None
    rows = list(csv.DictReader(open(f[0])))
    return rows[-1] if rows else None


def best_json(out):
    f = glob.glob(os.path.join(out, 'seed*', 'ckpt', '*.best.json'))
    return json.load(open(f[0])) if f else None


def wins(r):
    return r is not None and float(r['ratio']) < 1.0 and float(r['p_exact']) < 0.01


def epoch_val_ler(out):
    """{epoch: val_ler} from the run log ('[gru] epoch N: val_loss=... val_ler=...')."""
    f = glob.glob(os.path.join(out, 'seed*', 'run.log'))
    if not f:
        return {}
    txt = open(f[0], errors='replace').read()
    return {int(e): float(v) for e, v in re.findall(r'\] epoch (\d+): val_loss=\S+ val_ler=([0-9.]+)', txt)}


def other_gpu_processes():
    """Compute processes on the GPU (pid, MiB). Called only when none of ours runs, so any
    process listed belongs to someone else."""
    out = sh('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits')
    return [l.strip() for l in out.splitlines() if l.strip()]


def host_avail_gb():
    for l in open('/proc/meminfo'):
        if l.startswith('MemAvailable'):
            return int(l.split()[1]) // 1048576
    return 0


def resolve(job, stab, best_so_far):
    """Concrete settings of a row: STAB fields from stability.json, 9-10 from the best run."""
    j = dict(job)
    if j['cond'] == 'replicate':
        if best_so_far is None:
            return None, 'no finished run to replicate yet'
        src = best_so_far
        for k in ('peak_lr', 'warmup', 'clip', 'epochs', 'batch', 'data', 'ema', 'warm_start',
                  'units'):
            j[k] = src[k]
        j['note'] = f"replicate of {src['name']} (seed 1)"
    if 'STAB' in (j['peak_lr'] + j['warmup'] + j['clip']):
        if stab is None:
            return None, 'needs the Tier A promotion'
        if j['peak_lr'] == 'STAB':
            j['peak_lr'] = stab['peak_lr']
        elif j['peak_lr'] == 'STAB/sqrt2':
            j['peak_lr'] = f"{float(stab['peak_lr']) / math.sqrt(2):.6g}"
        if j['warmup'] == 'STAB':
            j['warmup'] = stab['warmup']
        if j['clip'] == 'STAB':
            j['clip'] = stab['clip']
    return j, ''


def launch_env(j, out, stamp):
    seeds, counts = DATA[j['data']]
    env = dict(D=D, ROUNDS=D, SEEDS=j['seed'], UNITS=j['units'], P=0.004, POOL=POOL,
               NTRAIN=15_000_000, BATCH=j['batch'], EPOCHS=j['epochs'], LR=j['peak_lr'],
               LR_SCHEDULE='cosine', LR_END='3e-5', GLOBAL_CLIPNORM=j['clip'],
               LR_WARMUP_EPOCHS=j['warmup'], MAX_PARAMS=MAX_PARAMS, EARLY_STOP=0,
               RESET_AFTER=0, UNROLL=0, EVAL_NO_TF32=1, GPU_MEM_MIB=4000,
               MIN_FREE_MIB=4000, OUT=out, STAMP=stamp, MALLOC_ARENA_MAX=2,
               EXTRA_POOL=' '.join(TRAINONLY.format(s) for s in seeds),
               EXTRA_N=' '.join(str(c) for c in counts))
    if j['ema'] != '-':
        env['EMA_DECAY'] = j['ema']
    if j['warm_start'] != '-':
        src = sorted(glob.glob(WARM_START[j['warm_start']]))
        if not src and not DRY:
            raise SystemExit(f"warm-start source {j['warm_start']} not found")
        env['WARM_START'] = src[-1] if src else WARM_START[j['warm_start']] + ' (not found here)'
    return env


def criterion(j, r, killed_note=''):
    parts = ['min val p_L', f"warmup {j['warmup']} ep", f"clipnorm {j['clip']}"]
    if j['ema'] != '-':
        parts.append(f"EMA {j['ema']} (EMA weights scored)")
    if j['warm_start'] != '-':
        parts.append(f"warm start from {j['warm_start']} (input kernel fresh)")
    parts.append(f"queue {j['name']}")
    if r is not None:
        parts.append(f"McNemar vs MWPM: b={r['student_only']} c={r['mwpm_only']} "
                     f"n_disc={r['n_discordant']} p={float(r['p_exact']):.3g}")
    if killed_note:
        parts.append(killed_note)
    return '; '.join(parts)


def write_result(j, out, done):
    r = eval_row(out) if done == 'Yes' else None
    b = best_json(out)
    logf = glob.glob(os.path.join(out, 'seed*', 'run.log'))
    txt = open(logf[0], errors='replace').read() if logf else ''
    pm = re.search(r'params=([\d,]+)', txt)
    t0 = os.path.getmtime(os.path.join(out, 'command.txt'))
    t1 = max([os.path.getmtime(f) for f in glob.glob(os.path.join(out, '**'), recursive=True)
              if os.path.isfile(f)] or [t0])
    note = open(os.path.join(out, 'KILLED')).read().strip() if done.startswith('Killed') else ''
    if done == 'Diverged':
        note = open(os.path.join(out, 'DIVERGED')).read().strip()
    row = [done, f"u{j['units']}", pm.group(1).replace(',', '') if pm else '', D, D,
           'cosine to 3e-5', j['peak_lr'], j['data'], j['seed'], 0.004, j['batch'], j['epochs'],
           b['best_epoch'] if b else '', r['mwpm_p_L'] if r else '', r['p_L'] if r else '',
           (round(float(r['p_L']) / float(r['mwpm_p_L']), 3) if r else ''),
           round(b['val_loss'], 5) if b else '', 'FALSE', criterion(j, r, note),
           round((t1 - t0) / 3600, 2), now()[:10]]
    new = not os.path.exists(RESULTS)
    with open(RESULTS, 'a', newline='') as fh:
        w = csv.writer(fh, delimiter='\t')
        if new:
            w.writerow(CSV_COLS)
        w.writerow(row)


def run_job(j, threshold, dry):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = os.path.join(DIR, f"{j['name']}_{stamp}")
    env = launch_env(j, out, stamp)
    cname = PREFIX + j['name']
    cmd = (['podman', 'run', '--rm', '--name', cname, '--security-opt=label=disable',
            '--device', 'nvidia.com/gpu=all', '-v', f'{HOME}:{HOME}', '-e', f'HOME={HOME}',
            '-w', f'{REPO}/src', IMG, 'env'] + [f'{k}={v}' for k, v in env.items()]
           + ['bash', 'gh200_run_student_scaling.sh'])
    if dry:
        log(f"DRY   {j['name']}: " + ' '.join(f'{k}={v}' for k, v in env.items()
                                             if k not in ('OUT', 'STAMP', 'HOME')))
        return 'dry'
    os.makedirs(out)
    with open(os.path.join(out, 'command.txt'), 'w') as fh:
        fh.write(' '.join(cmd) + '\n')
    with open(os.path.join(out, 'code_commit.txt'), 'w') as fh:
        fh.write(f"at_start repo_head {sh(f'git -C {REPO} rev-parse HEAD').strip()}\n")
    log(f"START {j['name']} lr={j['peak_lr']} warmup={j['warmup']} clip={j['clip']} "
        f"epochs={j['epochs']} batch={j['batch']} data={j['data']} seed={j['seed']} "
        f"ema={j['ema']} warm_start={j['warm_start']} -> {out}")
    with open(os.path.join(out, 'podman.log'), 'w') as pl:
        proc = subprocess.Popen(cmd, stdout=pl, stderr=subprocess.STDOUT)
    k_epoch = max(1, round(KILL_FRACTION * int(j['epochs'])))
    checked = False
    while proc.poll() is None:
        time.sleep(POLL_S)
        logf = glob.glob(os.path.join(out, 'seed*', 'run.log'))
        txt = open(logf[0], errors='replace').read() if logf else ''
        if re.search(r'loss: nan|loss=nan', txt):
            ep = len(re.findall(r'ms/step', txt))
            last = re.findall(r' - loss: ([0-9.e+-]+)', txt)
            sh(f'podman stop -t 30 {cname}')
            open(os.path.join(out, 'DIVERGED'), 'w').write(
                f"training loss nan at epoch {ep}; last finite epoch loss {last[-1] if last else 'none'}")
            log(f"DIVERGED {j['name']} at epoch {ep}")
            break
        vl = epoch_val_ler(out)
        if not checked and k_epoch in vl:
            checked = True
            if vl[k_epoch] > threshold:
                sh(f'podman stop -t 30 {cname}')
                open(os.path.join(out, 'KILLED'), 'w').write(
                    f"Killed @ ep {k_epoch}: val p_L {vl[k_epoch]:.6f} > {KILL_FACTOR} x baseline "
                    f"{threshold / KILL_FACTOR:.6f} = {threshold:.6f}")
                log(f"KILLED {j['name']} at epoch {k_epoch}: val p_L {vl[k_epoch]:.6f} > {threshold:.6f}")
                break
            log(f"CHECK {j['name']} epoch {k_epoch}: val p_L {vl[k_epoch]:.6f} <= {threshold:.6f}; continues")
    rc = proc.wait()
    open(os.path.join(out, 'exit_code'), 'w').write(f'{rc}\n')
    st = status(j['name'])
    done = {'done': 'Yes', 'diverged': 'Diverged'}.get(st, 'No')
    if st == 'killed':
        done = f'Killed @ ep {k_epoch}'
    write_result(j, out, done)
    r = eval_row(out) if st == 'done' else None
    log(f"END   {j['name']} {st}" + (f": p_L {r['p_L']} MWPM {r['mwpm_p_L']} ratio {r['ratio']} "
                                    f"McNemar b={r['student_only']} c={r['mwpm_only']} "
                                    f"p={float(r['p_exact']):.3g}{'  WIN' if wins(r) else ''}"
                                    if r else ''))
    return st


def skip(j, reason, dry):
    log(f"SKIP  {j['name']}: {reason}")
    if dry:
        return
    out = os.path.join(DIR, f"{j['name']}_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    os.makedirs(out)
    open(os.path.join(out, 'SKIPPED'), 'w').write(reason + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline-history', required=True,
                    help='history.json of the baseline run R02 (d9 u280 100M, 80 epochs)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    global DRY
    DRY = a.dry_run
    hb = json.load(open(a.baseline_history))
    base_k = round(KILL_FRACTION * len(hb['val_ler']))
    threshold = KILL_FACTOR * hb['val_ler'][base_k - 1]
    log(f"BEGIN repo={sh(f'git -C {REPO} rev-parse --short HEAD').strip()} baseline val p_L at "
        f"epoch {base_k} = {hb['val_ler'][base_k - 1]:.6f}; kill threshold {threshold:.6f}"
        + ('  (dry run)' if a.dry_run else ''))
    stacked_ok = 'nargs' in sh(f"grep -n \"'--units'\" {REPO}/src/eval_student_on_tail.py")
    dry_stab = None
    while True:
        jobs = read_jobs()
        by = {j['name']: j for j in jobs}
        stab = json.load(open(STAB)) if os.path.exists(STAB) else dry_stab
        # Promotion after Tier A.
        tier_a = [j for j in jobs if j['tier'] == 'A']
        if stab is None and all(status(j['name']) in ('done', 'killed', 'diverged', 'failed')
                                for j in tier_a):
            fin = [(float(eval_row(run_dir(j['name']))['ratio']), j) for j in tier_a
                   if status(j['name']) == 'done']
            if not fin:
                log('STOP  every Tier A run was stopped; no stability setting to promote')
                return
            ratio, w = min(fin, key=lambda t: t[0])
            b = best_json(run_dir(w['name']))
            stab = dict(source=w['name'], ratio=ratio, peak_lr=w['peak_lr'], warmup=w['warmup'],
                        clip=w['clip'], epochs=w['epochs'], best_epoch=b['best_epoch'])
            json.dump(stab, open(STAB, 'w'), indent=1)
            log(f"PROMOTE {w['name']} (ratio {ratio}): peak_lr={w['peak_lr']} warmup={w['warmup']} "
                f"clip={w['clip']}; best epoch {b['best_epoch']} of {w['epochs']}")
        # Best finished config so far (for 9-10), and whether anything won.
        fin_all = [(float(eval_row(run_dir(j['name']))['ratio']), j) for j in jobs
                   if j['cond'] != 'replicate' and status(j['name']) == 'done']
        best_so_far = min(fin_all, key=lambda t: t[0])[1] if fin_all else None
        any_win = any(wins(eval_row(run_dir(j['name']))) for _, j in fin_all)
        pending = [j for j in jobs if status(j['name']) == '']
        if a.dry_run:
            pending = [j for j in jobs if j['name'] not in getattr(main, 'seen', set())]
        if not pending:
            log('END   queue finished')
            return
        rep = [j for j in pending if j['cond'] == 'replicate']
        order = (rep + [j for j in pending if j not in rep]) if (any_win and rep) else pending
        j = order[0]
        if j['cond'] == 'replicate' and len(pending) > 1 and not any_win:
            j = [x for x in pending if x['cond'] != 'replicate'][0]
        if a.dry_run:
            main.seen = getattr(main, 'seen', set()) | {j['name']}
            if j['tier'] == 'B' and dry_stab is None:
                t1 = by[tier_a[0]['name']]
                dry_stab = dict(source=t1['name'] + ' (dry run: assumed)', peak_lr=t1['peak_lr'],
                                warmup=t1['warmup'], clip=t1['clip'], best_epoch='?')
                stab = dry_stab
        if j['cond'] == 'stacked' and not stacked_ok:
            skip(j, 'stacked GRU needs the two-layer change in the evaluation script, not '
                    'approved/present', a.dry_run)
            continue
        if j['cond'] == 'last10pct':
            if stab is None:
                skip(j, 'no promoted run', a.dry_run)
                continue
            if stab['best_epoch'] != '?' and int(stab['best_epoch']) <= 0.9 * int(stab['epochs']):
                skip(j, f"promoted run {stab['source']} best epoch {stab['best_epoch']} of "
                        f"{stab['epochs']} is not in the last 10%", a.dry_run)
                continue
        rj, why = resolve(j, stab, best_so_far if not a.dry_run else by[tier_a[0]['name']])
        if rj is None:
            if a.dry_run:
                log(f"DRY   {j['name']}: {why}")
                continue
            log(f"WAIT  {j['name']}: {why}")
            time.sleep(POLL_S)
            continue
        if not a.dry_run:
            while True:
                procs = other_gpu_processes()
                need = int(rj['need_host_gb'])
                if not procs and host_avail_gb() >= need:
                    break
                log(f"WAIT  {rj['name']}: GPU processes {procs or 'none'}; host available "
                    f"{host_avail_gb()} GB (need {need}); re-check in {GPU_WAIT_S // 60} min")
                time.sleep(GPU_WAIT_S)
        run_job(rj, threshold, a.dry_run)


if __name__ == '__main__':
    main()
