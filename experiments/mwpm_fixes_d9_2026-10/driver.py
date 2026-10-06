#!/usr/bin/env python3
# Created: 2026-10-03
# Last modified: 2026-10-06 (a HOLD file in this folder stops new launches; its text is logged). Earlier 2026-10-05 (rows run concurrently when host memory allows; GPU-sharing wait removed: the GPU is not shared. Earlier the same day: run_job resumes a paused run). Earlier 2026-10-03 (amended before launch: kill rules replaced; 9-8 removed)
"""Queue driver for the d=9 MWPM-fix runs (prereg/mwpm_fixes_d9_2026-10.md).

One run on the GPU at a time, in the order of jobs.tsv, with the rules fixed in the
pre-registration:

  launch    (2026-10-05) rows run concurrently, each in its own thread: the next row starts
            when it can be resolved, host memory available >= its need_host_gb, no queue run
            is still loading its data, and swap in use has not grown by more than 2 GB over
            the last ~10 minutes. The GPU is not shared with other users, so the earlier
            wait for other users' GPU processes is gone (gpu_free_for_us is kept, unused).
  host      host memory available >= the job's need_host_gb
  budget    MAX_PARAMS = the single-layer u280 count at d=9 (303,521); the launcher and
            train_student.py refuse a model over it
  kill      training loss nan -> stopped (DIVERGED). Validation p_L above 1.5 x the run's
            own best so far for 5 consecutive epochs -> stopped (KILLED). Validation p_L at
            epoch 60 above 2 x the baseline run's (R02) validation p_L at epoch 60 -> stopped
            (KILLED). All recorded with the epoch.
  win       p_L / MWPM on the evaluation block < 1 with McNemar p_exact < 0.01
  promote   after 9-1, 9-2, 9-3 finish, the one with the lowest p_L / MWPM sets the peak LR,
            warmup and clipnorm of every later row marked STAB (stability.json). If all
            three were stopped, the driver stops here.
  9-6       only if the promoted run's best epoch is in the last 10% of its epochs;
            otherwise skipped, with the reason logged
  9-10      the best config so far (lowest ratio among the finished runs), seed 1. If a run
            wins, 9-10 runs next instead of last.

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
import threading
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
GPU_OTHER_MAX_MIB = 10 * 1024
GPU_IDLE_CHECK_S = 300
GPU_IDLE_MEAN = 5.0
GPU_IDLE_MAX = 20.0
BLOCKED = os.path.join(DIR, 'blocked.json')
POLL_S = 60
KILL_EPOCH = 60            # absolute epoch compared with the baseline
KILL_FACTOR = 2.0          # x baseline val p_L at KILL_EPOCH
STREAK_FACTOR = 1.5        # x the run's own best val p_L so far
STREAK_EPOCHS = 5          # consecutive epochs above it
CSV_COLS = ['Done?', 'Width', 'Parameters', 'd', 'r', 'LR Schedule', 'LR', 'Data', 'Seed', 'p',
            'Batch', 'Max Epoch', 'Best Ckpt', 'MWPM', 'p_L', 'Relative to MWPM', 'Val_loss',
            'reset_after', 'Checkpoint criterion', 'runtime (h)', 'Date']


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


DRY = False
LOCK = threading.Lock()             # log lines and results rows from concurrent runs


def log(msg):
    line = f'[mwpm-d9 {now()}] {msg}'
    with LOCK:
        print(line, flush=True)
        if not DRY:                   # a dry run writes no file
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


def streak_reason(vl):
    """(epoch, reason) when validation p_L has been above STREAK_FACTOR x the run's own best
    so far for STREAK_EPOCHS consecutive epochs, else None."""
    best, run = None, 0
    for e in sorted(vl):
        v = vl[e]
        best = v if best is None else min(best, v)
        run = run + 1 if v > STREAK_FACTOR * best else 0
        if run >= STREAK_EPOCHS:
            return e, (f"val p_L above {STREAK_FACTOR:g} x the run's best so far ({best:.6f}) for "
                       f"{STREAK_EPOCHS} consecutive epochs, epochs {e - STREAK_EPOCHS + 1}-{e} "
                       f"(last {v:.6f})")
    return None


def other_gpu_processes():
    """Compute processes on the GPU as (pid, MiB, user). Called only when none of ours runs."""
    out = sh('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits')
    procs = []
    for l in out.splitlines():
        if not l.strip():
            continue
        pid, mib = [x.strip() for x in l.split(',')[:2]]
        user = sh(f'ps -o user= -p {pid}').strip() or '?'
        procs.append((pid, int(float(mib)), user))
    return procs


def gpu_free_for_us():
    """(ok, note). ok when no other compute process, or when the others hold < 10 GB in total
    and GPU utilization stays near 0% over a 5-minute check."""
    procs = other_gpu_processes()
    if not procs:
        return True, 'no other GPU process'
    mib = sum(p[1] for p in procs)
    desc = ', '.join(f'pid {p[0]} ({p[2]}, {p[1]} MiB)' for p in procs)
    if mib >= GPU_OTHER_MAX_MIB:
        return False, f'{desc}: {mib} MiB >= {GPU_OTHER_MAX_MIB} MiB'
    util = []
    for _ in range(GPU_IDLE_CHECK_S // 10):
        u = sh('nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits').split()
        util.append(float(u[0]) if u else 100.0)
        time.sleep(10)
    mean, peak = sum(util) / len(util), max(util)
    ok = mean < GPU_IDLE_MEAN and peak < GPU_IDLE_MAX
    return ok, (f'{desc}; utilization over {GPU_IDLE_CHECK_S // 60} min mean {mean:.1f}% max '
                f'{peak:.0f}% -> {"idle, launching alongside" if ok else "busy"}')


def swap_used_gb():
    for l in open('/proc/meminfo'):
        if l.startswith('SwapTotal'):
            tot = int(l.split()[1])
        elif l.startswith('SwapFree'):
            free = int(l.split()[1])
    return (tot - free) / 1048576


def queue_run_loading(jobs):
    """Name of a queue run whose container runs but whose log has no epoch line yet."""
    up = sh("podman ps --format '{{.Names}}'").split()
    for j in jobs:
        if PREFIX + j['name'] in up:
            out = run_dir(j['name'])
            f = glob.glob(os.path.join(out, 'seed*', 'run.log')) if out else []
            if not f or not re.search(r'\] epoch \d+: val_loss', open(f[0], errors='replace').read()):
                return j['name']
    return None


def add_blocked(kind, seconds):
    b = json.load(open(BLOCKED)) if os.path.exists(BLOCKED) else {}
    b[kind] = b.get(kind, 0) + seconds
    json.dump(b, open(BLOCKED, 'w'), indent=1)


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
    with LOCK:
        new = not os.path.exists(RESULTS)
        with open(RESULTS, 'a', newline='') as fh:
            w = csv.writer(fh, delimiter='\t')
            if new:
                w.writerow(CSV_COLS)
            w.writerow(row)


def prepare_resume(src, out):
    """Set up `out` to continue the paused run in `src` (as resume_paused.sh did for the
    earlier pauses): .best.* into out/seed0/ckpt, the last-epoch weights (model + optimizer
    state, which carries the update count and so the LR-schedule step) and the history of
    epochs 1..N rebuilt from run.log into out/resume_from. Returns (resume env, N, earlier
    val p_L by epoch)."""
    import shutil
    sd = os.path.join(src, 'seed0')
    txt = open(os.path.join(sd, 'run.log'), errors='replace').read()
    lines = [l for l in txt.splitlines() if 'ms/step' in l]
    n = len(lines)
    tag = os.path.basename(glob.glob(os.path.join(sd, 'ckpt', '*.best.json'))[0])[:-len('.best.json')]
    ck, rf = os.path.join(out, 'seed0', 'ckpt'), os.path.join(out, 'resume_from')
    os.makedirs(ck)
    os.makedirs(rf)
    for ext in ('.best.keras', '.best.weights.h5', '.best.json'):
        shutil.copy2(os.path.join(sd, 'ckpt', tag + ext), ck)
    shutil.copy2(os.path.join(sd, 'ckpt', tag + '.lastepoch.weights.h5'), rf)
    h = {}
    for l in lines:
        for k, v in re.findall(r' - ([a-z_]+): ([-0-9.e+]+)', l):
            h.setdefault(k, []).append(float(v))
    if not all(len(v) == n for v in h.values()):
        raise SystemExit('history rebuild: unequal lengths ' + str({k: len(v) for k, v in h.items()}))
    json.dump(h, open(os.path.join(rf, tag + '.history.json'), 'w'))
    env = dict(RESUME_WEIGHTS=os.path.join(rf, tag + '.lastepoch.weights.h5'), INITIAL_EPOCH=n,
               RESUME_BEST_JSON=os.path.join(ck, tag + '.best.json'),
               RESUME_HISTORY=os.path.join(rf, tag + '.history.json'))
    return env, n, epoch_val_ler(src)


def run_job(j, threshold, dry, resume_from=None):
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
    prior, n_done = {}, 0
    if resume_from:
        renv, n_done, prior = prepare_resume(resume_from, out)
        env.update(renv)
        cmd = cmd[:cmd.index('bash')] + [f'{k}={v}' for k, v in renv.items()] + cmd[cmd.index('bash'):]
        log(f"RESUME {j['name']} from {resume_from} after epoch {n_done}")
    with open(os.path.join(out, 'command.txt'), 'w') as fh:
        fh.write(' '.join(cmd) + '\n')
    with open(os.path.join(out, 'code_commit.txt'), 'w') as fh:
        fh.write(f"at_start repo_head {sh(f'git -C {REPO} rev-parse HEAD').strip()}\n")
    log(f"START {j['name']} lr={j['peak_lr']} warmup={j['warmup']} clip={j['clip']} "
        f"epochs={j['epochs']} batch={j['batch']} data={j['data']} seed={j['seed']} "
        f"ema={j['ema']} warm_start={j['warm_start']} -> {out}")
    with open(os.path.join(out, 'podman.log'), 'w') as pl:
        proc = subprocess.Popen(cmd, stdout=pl, stderr=subprocess.STDOUT)
    checked = n_done >= KILL_EPOCH      # a resumed run past epoch 60 was checked before
    k_epoch = None
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
        vl = {**prior, **epoch_val_ler(out)}     # epochs before a pause count too
        reason = streak_reason(vl)
        if reason is None and not checked and KILL_EPOCH in vl:
            checked = True
            if vl[KILL_EPOCH] > threshold:
                reason = (KILL_EPOCH, f"val p_L {vl[KILL_EPOCH]:.6f} at epoch {KILL_EPOCH} > "
                          f"{KILL_FACTOR:g} x baseline {threshold / KILL_FACTOR:.6f} = {threshold:.6f}")
            else:
                log(f"CHECK {j['name']} epoch {KILL_EPOCH}: val p_L {vl[KILL_EPOCH]:.6f} <= "
                    f"{threshold:.6f}; continues")
        if reason is not None:
            k_epoch = reason[0]
            sh(f'podman stop -t 30 {cname}')
            open(os.path.join(out, 'KILLED'), 'w').write(f"Killed @ ep {k_epoch}: {reason[1]}")
            log(f"KILLED {j['name']} at epoch {k_epoch}: {reason[1]}")
            break
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
    threshold = KILL_FACTOR * hb['val_ler'][KILL_EPOCH - 1]
    log(f"BEGIN repo={sh(f'git -C {REPO} rev-parse --short HEAD').strip()} baseline val p_L at "
        f"epoch {KILL_EPOCH} = {hb['val_ler'][KILL_EPOCH - 1]:.6f}; epoch-{KILL_EPOCH} kill "
        f"threshold {threshold:.6f}; streak rule {STREAK_FACTOR:g} x own best for "
        f"{STREAK_EPOCHS} epochs" + ('  (dry run)' if a.dry_run else ''))
    dry_stab = None
    threads = {}                       # rows started by this driver, run in threads
    swap_hist = []
    last_wait = None
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
        pending = [j for j in jobs if status(j['name']) == '' and j['name'] not in threads]
        if a.dry_run:
            pending = [j for j in jobs if j['name'] not in getattr(main, 'seen', set())]
        if not pending:
            if any(t.is_alive() for t in threads.values()):
                time.sleep(POLL_S)
                continue
            log('END   queue finished')
            return
        rep = [j for j in pending if j['cond'] == 'replicate']
        order = (rep + [j for j in pending if j not in rep]) if (any_win and rep) else pending
        j = order[0]
        if j['cond'] == 'replicate' and len(pending) > 1 and not any_win:
            j = [x for x in pending if x['cond'] != 'replicate'][0]
        # 9-10 replicates the best of all the other rows, so (unless a run won) it waits until
        # none of them is still running.
        if (j['cond'] == 'replicate' and not any_win and not a.dry_run
                and any(status(x['name']) == 'running' for x in jobs if x['cond'] != 'replicate')):
            msg = f"WAIT  {j['name']}: replicate waits until every other row has finished"
            if msg != last_wait:
                log(msg)
                last_wait = msg
            time.sleep(POLL_S)
            continue
        if a.dry_run:
            main.seen = getattr(main, 'seen', set()) | {j['name']}
            if j['tier'] == 'B' and dry_stab is None:
                t1 = by[tier_a[0]['name']]
                dry_stab = dict(source=t1['name'] + ' (dry run: assumed)', peak_lr=t1['peak_lr'],
                                warmup=t1['warmup'], clip=t1['clip'], best_epoch='?')
                stab = dry_stab
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
            msg = f"WAIT  {j['name']}: {why}"
            if msg != last_wait:           # log a waiting reason once, not every minute
                log(msg)
                last_wait = msg
            time.sleep(POLL_S)
            continue
        if a.dry_run:
            run_job(rj, threshold, True)
            continue
        # HOLD: no new launch while this file exists (runs already going are not touched).
        hold = os.path.join(DIR, 'HOLD')
        if os.path.exists(hold):
            msg = f"WAIT  {rj['name']}: HOLD ({open(hold).read().strip()[:200]})"
            if msg != last_wait:
                log(msg)
                last_wait = msg
            time.sleep(POLL_S)
            continue
        # Concurrent launch, when memory allows and no other queue run is loading.
        sw = swap_used_gb()
        swap_hist = (swap_hist + [sw])[-10:]
        need = int(rj['need_host_gb'])
        loading = queue_run_loading(jobs)
        why = ('' if host_avail_gb() >= need else f'host available {host_avail_gb()} GB < {need}')
        why = why or (f'{loading} still loading' if loading else '')
        why = why or (f'swap grew {sw - min(swap_hist):.1f} GB' if sw - min(swap_hist) > 2 else '')
        if why:
            msg = f"WAIT  {rj['name']}: {why}"
            if msg != last_wait:
                log(msg)
                last_wait = msg
            time.sleep(POLL_S)
            continue
        last_wait = None
        log(f"LAUNCH {rj['name']} alongside {sorted(n for n, t in threads.items() if t.is_alive()) or 'nothing of this driver'}; "
            f"host available {host_avail_gb()} GB (need {need}), swap {sw:.0f} GB")
        t = threading.Thread(target=run_job, args=(rj, threshold, False), name=rj['name'])
        t.start()
        threads[rj['name']] = t
        time.sleep(POLL_S)


if __name__ == '__main__':
    main()
