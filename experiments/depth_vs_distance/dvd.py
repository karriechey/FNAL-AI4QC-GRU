# Created: 2026-10-28
# Last modified: 2026-10-31

"""Depth vs distance (PLAN.md): run geometry, log parsing across resumed parts, kill rule,
LR rule, L_min, results CSV and STATUS.md.

python dvd.py info <run>           shell assignments for run_dvd.sh
python dvd.py kill-check <folder>  exit 3 and KILLED/DIVERGED when the rule fires
python dvd.py lr-rule <d>          lr/d<d>.json
python dvd.py lmin <d>             lmin/d<d>.json
python dvd.py report               GRU_depth_vs_distance_runs.csv, STATUS.md
python dvd.py plan-table           run table with cost estimates (markdown)
"""
import csv
import datetime
import glob
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'experiments', 'mwpm_fixes_d9_2026-10'))
sys.path.insert(0, os.path.join(REPO, 'experiments', 'depth_scaling_d7_2026-10'))

DISTANCES = (5, 7, 9)
DEPTHS = (1, 2, 3, 4, 5)
SCAN_DEPTHS = (1, 5)
SCAN_LRS = ('1e-3', '2e-3', '3e-3')
FALLBACK_LR = '5e-4'          # lrscan_d7's conditional run, when no grid LR is stable at both depths
SCAN_EPOCHS = 20
EPOCHS = {5: 80, 7: 80, 9: 160}
FSB = {(5, 0): 6 * 10**12, (7, 0): 7 * 10**12, (9, 0): 8 * 10**12,
       (5, 1): 9 * 10**12, (7, 1): 10 * 10**12, (9, 1): 11 * 10**12}
P_SIG = 0.01
UNSTABLE_FACTOR = 1.10        # lrscan_d7: train loss above 1.10 x running min
UNSTABLE_EPOCHS = 3           # for 3 or more epochs
STEPS_PER_EPOCH = 10000       # 100M shots / batch 10000

# runtime model for the plan and the ETA: s/epoch = T1[d] x (1 + 0.12 (L - 1)), from the d5/d7/d9
# fresh-shot history (f5w 230, f7w 355, f2280/f3280 610-640 s/epoch with 2-3 runs on the card)
# scaled up for 5-8 concurrent runs; plus selection and eval scoring
T1 = {5: 350, 7: 550, 9: 800}
SCORE_H = {5: 0.5, 7: 1.0, 9: 4.0}


def det_per_round(d):
    return d * d - 1


def width(d):
    return round(3.5 * det_per_round(d))


def n_params(d, L):
    u, n_in = width(d), det_per_round(d)
    return 3 * (n_in * u + u * u + u) + 3 * (2 * u * u + u) * (L - 1) + u + 1


def est_hours(d, L, epochs, scored):
    return T1[d] * (1 + 0.12 * (L - 1)) * epochs / 3600 + (SCORE_H[d] if scored else 0)


def parse(run):
    m = re.fullmatch(r'lrscan_d(\d)_L(\d\d)_lr(\S+)', run)
    if m:
        d, L, lr = int(m.group(1)), int(m.group(2)), m.group(3)
        if d in DISTANCES and L in SCAN_DEPTHS and lr in SCAN_LRS + (FALLBACK_LR,):
            return dict(kind='scan', part='A', d=d, L=L, seed=0, lr=lr)
    m = re.fullmatch(r'dvd_d(\d)_L(\d\d)_u(\d+)(_s1)?', run)
    if m:
        d, L, w = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if d in DISTANCES and L in DEPTHS and w == width(d):
            seed = 1 if m.group(4) else 0
            return dict(kind='sweep', part='C' if seed else 'B', d=d, L=L, seed=seed, lr=None)
    raise SystemExit(f'unknown run {run}')


def scan_names(d, lrs=SCAN_LRS):
    return [f'lrscan_d{d}_L{L:02d}_lr{lr}' for L in (5, 1) for lr in lrs]


def sweep_name(d, L, seed=0):
    return f'dvd_d{d}_L{L:02d}_u{width(d)}' + ('_s1' if seed else '')


def newest(run):
    found = sorted(glob.glob(os.path.join(HERE, 'runs', f'{run}_2*')))
    return found[-1] if found else None


def paused_distances():
    f = os.path.join(HERE, 'queue', 'paused_d')
    return {int(x) for x in open(f).read().split()} if os.path.exists(f) else set()


def run_distance(run):
    return int(re.match(r'(?:lrscan|dvd)_d(\d+)_', run).group(1))


def marker(out, *names):
    return next((n for n in names if os.path.exists(os.path.join(out, n))), None)


def finished(run):
    # DONE, KILLED or DIVERGED in the newest folder (FAILED is retried by the queue)
    out = newest(run)
    return bool(out) and marker(out, 'DONE', 'KILLED', 'DIVERGED') is not None


def seed_dir(out):
    return glob.glob(os.path.join(out, 'seed[0-9]'))[0] if glob.glob(os.path.join(out, 'seed[0-9]')) else os.path.join(out, 'seed0')


def logs(out):
    sd = seed_dir(out)
    parts = sorted(glob.glob(os.path.join(sd, 'run_part*.log')), key=lambda f: int(re.search(r'part(\d+)', f).group(1)))
    cur = os.path.join(sd, 'run.log')
    texts = [open(f, errors='replace').read() for f in parts]
    if os.path.exists(cur):
        t = open(cur, errors='replace').read()
        if not texts or t != texts[-1]:
            texts.append(t)
    return texts


def epochs_of(out):
    # {epoch: dict(loss, val_ler, s)} over every part; a later part overrides a repeated epoch
    ep = {}
    for txt in logs(out):
        cur = None
        for line in txt.splitlines():
            m = re.match(r'^\[gru\] epoch (\d+): val_loss=\S+ val_ler=([0-9.]+)', line)
            if m:
                cur = int(m.group(1))
                ep.setdefault(cur, {})['val_ler'] = float(m.group(2))
                continue
            m = re.match(r'^\[gru\] epoch (\d+): global grad norm', line)
            if m:
                cur = int(m.group(1))
                continue
            m = re.match(r'^\d+/\d+ - (\d+)s - loss: (\S+)', line)
            if m and cur is not None:
                ep.setdefault(cur, {}).update(loss=float(m.group(2)), s=int(m.group(1)))
                cur = None
    return {e: v for e, v in ep.items() if 'loss' in v}


def kill_check(out):
    from driver import streak_reason
    ep = epochs_of(out)
    if any(math.isnan(v['loss']) for v in ep.values()) or any(re.search(r'loss: nan', t) for t in logs(out)):
        e = max(ep) if ep else 0
        open(os.path.join(out, 'DIVERGED'), 'w').write(f'training loss nan at epoch {e}\n')
        print(f'DIVERGED epoch {e}')
        return 3
    r = streak_reason({e: v['val_ler'] for e, v in ep.items() if 'val_ler' in v})
    if r is not None:
        open(os.path.join(out, 'KILLED'), 'w').write(f'Killed @ ep {r[0]}: {r[1]}\n')
        print(f'KILLED epoch {r[0]}: {r[1]}')
        return 3
    return 0


def longest_rise(losses):
    best, run, longest, worst = math.inf, 0, 0, 0.0
    for v in losses:
        if best < math.inf:
            worst = max(worst, v / best)
        run = run + 1 if v > UNSTABLE_FACTOR * best else 0
        longest = max(longest, run)
        best = min(best, v)
    return longest, worst


def classify(out):
    ep = epochs_of(out)
    losses = [ep[e]['loss'] for e in sorted(ep)]
    longest, worst = longest_rise(losses)
    note = ''
    for f in ('KILLED', 'DIVERGED'):
        if os.path.exists(os.path.join(out, f)):
            note = open(os.path.join(out, f)).read().strip()
    if note or any(math.isnan(v) for v in losses):
        cls = 'diverged'
    elif longest >= UNSTABLE_EPOCHS:
        cls = 'unstable'
    else:
        cls = 'stable'
    return dict(cls=cls, epochs=len(losses), longest=longest, worst=round(worst, 3), note=note,
                final_loss=losses[-1] if losses else None)


def lr_rule(d):
    # highest LR stable at both L = 1 and L = 5; 5e-4 at both depths when none is
    os.makedirs(os.path.join(HERE, 'lr'), exist_ok=True)
    status = {}
    for run in scan_names(d, SCAN_LRS + (FALLBACK_LR,)):
        if finished(run):
            status[run] = classify(newest(run))['cls']
    missing = [r for r in scan_names(d) if r not in status]
    if missing:
        rule = dict(lr=None, reason=f'scan runs not finished: {missing}')
    else:
        ok = [lr for lr in SCAN_LRS if all(status[f'lrscan_d{d}_L{L:02d}_lr{lr}'] == 'stable' for L in SCAN_DEPTHS)]
        fb = [f'lrscan_d{d}_L{L:02d}_lr{FALLBACK_LR}' for L in (5, 1)]
        if ok:
            lr = max(ok, key=float)
            rule = dict(lr=lr, reason=f'highest LR stable at L = 1 and L = 5 (stable: {ok})')
        elif any(r not in status for r in fb):
            rule = dict(lr=None, need=[r for r in fb if r not in status],
                        reason=f'no LR in {list(SCAN_LRS)} stable at both depths; {FALLBACK_LR} runs needed')
        elif all(status[r] == 'stable' for r in fb):
            rule = dict(lr=FALLBACK_LR, reason=f'no LR in {list(SCAN_LRS)} stable at both depths; {FALLBACK_LR} stable')
        else:
            rule = dict(lr=None, stop=True, reason=f'no LR stable at both depths, {FALLBACK_LR} included')
    rule['status'] = status
    json.dump(rule, open(os.path.join(HERE, 'lr', f'd{d}.json'), 'w'), indent=1)
    print(json.dumps(rule))


def sweep_lr(d):
    f = os.path.join(HERE, 'lr', f'd{d}.json')
    return json.load(open(f)).get('lr') if os.path.exists(f) else None


def eval_scores(run):
    # (ratio, mcnemar dict, per-shot dump, mwpm dump) for a DONE sweep run
    from helpers import mcnemar_from_correct
    from pair_two_decoders import load_dump
    out = newest(run)
    if not out or marker(out, 'DONE') is None:
        return None
    dump = os.path.join(out, 'eval_selected_per_shot.npz')
    g = load_dump(dump)
    m = load_dump(dump, 'mwpm_correct')
    err, m_err = int((~g[1]).sum()), int((~m[1]).sum())
    return dict(ratio=err / m_err, p_L=err / len(g[1]), mwpm=m_err / len(g[1]),
                mc=mcnemar_from_correct(g[1], m[1]), g=g, m=m)


def beats_mwpm(s):
    return s is not None and s['ratio'] < 1 and s['mc']['p_exact'] < P_SIG


def lmin(d):
    os.makedirs(os.path.join(HERE, 'lmin'), exist_ok=True)
    scores = {L: eval_scores(sweep_name(d, L)) for L in DEPTHS}
    open_ = [L for L in DEPTHS if not finished(sweep_name(d, L))]
    if open_:
        res = dict(lmin=None, seed_runs=[], reason=f'L = {open_} not finished')
    else:
        win = [L for L in DEPTHS if beats_mwpm(scores[L])]
        if win:
            lm = win[0]
            res = dict(lmin=lm, seed_runs=[sweep_name(d, L, 1) for L in (lm - 1, lm) if L >= 1],
                       reason=f'smallest L with ratio < 1 and McNemar p < {P_SIG}')
        else:
            res = dict(lmin=None, seed_runs=[], reason=f'no L in 1-5 beats MWPM with p < {P_SIG}')
    res['ratios'] = {L: (round(s['ratio'], 4) if s else None) for L, s in scores.items()}
    res['p'] = {L: (s['mc']['p_exact'] if s else None) for L, s in scores.items()}
    json.dump(res, open(os.path.join(HERE, 'lmin', f'd{d}.json'), 'w'), indent=1)
    print(json.dumps(res))


def info(run):
    import partitions as P
    r = parse(run)
    d, L = r['d'], r['L']
    lr = r['lr'] or sweep_lr(d) or ''
    ev0, ev1 = P.block('eval')
    rounds_width_layers = d * width(d) * L
    gpu_mib = max(4000, int(math.ceil((2000 + 0.8 * rounds_width_layers) / 1000) * 1000))
    v = dict(KIND=r['kind'], PART=r['part'], D=d, L=L, W=width(d), SEED=r['seed'], LRV=lr,
             EPOCHS=EPOCHS[d], FSB=FSB[(d, r['seed'])], WANT=n_params(d, L),
             STOPS=SCAN_EPOCHS * STEPS_PER_EPOCH if r['kind'] == 'scan' else '',
             EVAL_START=ev0, N_EVAL=ev1 - ev0, GPU_MEM_MIB=gpu_mib,
             NEED_HOST_GB=60 if d == 9 else 30)
    print('\n'.join(f'{k}={v[k]}' for k in v))


# results table: GRU_depth_scaling_runs_2026-10-23 columns plus width/detectors and seed
COLS = ['Run', 'Part', 'L', 'Width', 'width/detectors', 'seed', 'Parameters', 'Peak LR', 'eta x N_det', 'Status',
        'Killed epoch', 'Sel x MWPM ep20', 'Sel x MWPM ep40', 'Sel x MWPM ep60', 'Sel x MWPM ep80',
        'Selected epoch', 'p_L', 'MWPM', 'Relative to MWPM', 'McNemar vs MWPM', 'McNemar vs comparison',
        'Final train loss', 'Last-20 slope', 'Last-20 t', 'Converged', 's/epoch', 'runtime (h)', 'Date', 'Notes']


def p_str(p):
    return '<1e-300' if p == 0 else f'{p:.2g}'


def mc_str(m):
    return f"b={m['rcnn_only']} c={m['mwpm_only']} n_disc={m['n_discordant']} p={p_str(m['p_exact'])}"


def runtime_h(out):
    t0 = datetime.datetime.strptime(re.search(r'_(\d{8}T\d{6}Z)$', out).group(1), '%Y%m%dT%H%M%SZ')
    t0 = t0.replace(tzinfo=datetime.timezone.utc).timestamp()
    ends = [os.path.join(out, m) for m in ('DONE', 'KILLED', 'DIVERGED', 'FAILED') if os.path.exists(os.path.join(out, m))]
    if not ends:
        # os.walk skips the seed<S>/epochs symlink, whose snapshots the /tmp keepalive touches every day
        ends = [os.path.join(root, f) for root, _, fs in os.walk(out) for f in fs
                if not os.path.islink(os.path.join(root, f))]
    return round((max(os.path.getmtime(f) for f in ends) - t0) / 3600, 2) if ends else ''


def all_runs():
    names = []
    for d in DISTANCES:
        names += scan_names(d, SCAN_LRS + (FALLBACK_LR,))
        names += [sweep_name(d, L) for L in DEPTHS]
    for d in DISTANCES:
        names += [sweep_name(d, L, 1) for L in DEPTHS]
    return names


def row_for(run, scores):
    from helpers import mcnemar_from_correct
    r = parse(run)
    out = newest(run)
    d, L = r['d'], r['L']
    env = dict(re.findall(r'^(\w+)=(.*)$', open(os.path.join(out, 'env.txt')).read(), re.M)) \
        if os.path.exists(os.path.join(out, 'env.txt')) else {}
    lr = env.get('LR', r['lr'] or '')
    ep = epochs_of(out)
    secs = [ep[e]['s'] for e in sorted(ep)][1:]
    mk = marker(out, 'DONE', 'KILLED', 'DIVERGED', 'FAILED')
    c = classify(out)
    row = {'Run': run, 'Part': r['part'], 'L': L, 'Width': f'u{width(d)}',
           'width/detectors': round(width(d) / det_per_round(d), 2), 'seed': r['seed'],
           'Parameters': n_params(d, L), 'Peak LR': lr,
           'eta x N_det': round(float(lr) * det_per_round(d) * d, 2) if lr else '',
           'Final train loss': round(c['final_loss'], 5) if c['final_loss'] is not None else '',
           's/epoch': round(statistics.median(secs)) if secs else '',
           'runtime (h)': runtime_h(out), 'Date': datetime.date.today().isoformat()}
    notes = []
    killed = c['note']
    if killed:
        row['Killed epoch'] = re.search(r'(?:ep|epoch) (\d+)', killed).group(1)
        notes.append(killed)
    if r['kind'] == 'scan':
        row['Status'] = c['cls'] if mk in ('DONE', 'KILLED', 'DIVERGED') else (mk or 'running').lower()
        if row['Status'] == 'running' and d in paused_distances():
            row['Status'] = 'paused'
        notes.insert(0, f"{c['epochs']} epochs of the {EPOCHS[d]}-epoch schedule; longest rise {c['longest']} epochs, "
                        f"max loss / running min {c['worst']}")
    else:
        row['Status'] = {'DONE': 'Done', 'KILLED': 'killed', 'DIVERGED': 'diverged', 'FAILED': 'failed'}.get(mk, 'running')
        if row['Status'] == 'running' and d in paused_distances():
            row['Status'] = 'paused'
        sel_csv = os.path.join(out, 'selection_scores.csv')
        by_ep = {int(x['epoch']): float(x['ratio']) for x in csv.DictReader(open(sel_csv))} \
            if os.path.exists(sel_csv) else {}
        for e in (20, 40, 60, 80):
            row[f'Sel x MWPM ep{e}'] = by_ep.get(e, '')
        if EPOCHS[d] > 80 and EPOCHS[d] in by_ep:
            notes.append(f'selection block x MWPM at epoch {EPOCHS[d]}: {by_ep[EPOCHS[d]]}')
        s = scores.get(run)
        if s:
            sel = json.load(open(os.path.join(out, 'selected.json')))
            row.update({'Selected epoch': sel['epoch'], 'p_L': round(s['p_L'], 6), 'MWPM': round(s['mwpm'], 6),
                        'Relative to MWPM': round(s['ratio'], 4), 'McNemar vs MWPM': mc_str(s['mc']),
                        'Last-20 slope': sel['last20_slope_per_epoch'], 'Last-20 t': sel['last20_slope_t'],
                        'Converged': '' if sel['not_converged'] is None else ('no' if sel['not_converged'] else 'yes')})
            if sel['not_converged']:
                notes.append('still improving: last-20 selection slope negative with t < -2')
            other = sweep_name(d, L) if r['seed'] else (sweep_name(d, L - 1) if L > 1 else None)
            o = scores.get(other) if other else None
            if o:
                a, b = s['g'], o['g']
                if len(a[0]) == len(b[0]) and (a[0] == b[0]).all():
                    row['McNemar vs comparison'] = f'vs {other}: ' + mc_str(mcnemar_from_correct(a[1], b[1]))
    row['Notes'] = '; '.join(notes)
    return row


def eta_hours(run, out):
    r = parse(run)
    E = SCAN_EPOCHS if r['kind'] == 'scan' else EPOCHS[r['d']]
    if out is None:
        return est_hours(r['d'], r['L'], E, r['kind'] != 'scan')
    ep = epochs_of(out)
    secs = [ep[e]['s'] for e in sorted(ep)][1:]
    spe = statistics.median(secs) if secs else T1[r['d']] * (1 + 0.12 * (r['L'] - 1))
    left = max(0, E - (max(ep) if ep else 0)) * spe / 3600
    return left + (SCORE_H[r['d']] if r['kind'] != 'scan' else 0)


def report():
    scores = {}
    for run in all_runs():
        if parse(run)['kind'] == 'sweep':
            s = eval_scores(run)
            if s:
                scores[run] = s
    rows = [row_for(r, scores) for r in all_runs() if newest(r)]
    path = os.path.join(HERE, 'GRU_depth_vs_distance_runs.csv')
    with open(path + '.tmp', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in COLS})
    os.replace(path + '.tmp', path)

    now = datetime.datetime.now(datetime.timezone.utc)
    queued_file = os.path.join(HERE, 'queue', 'planned.txt')
    planned = open(queued_file).read().split() if os.path.exists(queued_file) else []
    done, running, failed, queued, paused = [], [], [], [], []
    for run in dict.fromkeys(planned + [r['Run'] for r in rows]):
        out = newest(run)
        mk = marker(out, 'DONE', 'KILLED', 'DIVERGED', 'FAILED') if out else None
        if mk in ('DONE', 'KILLED', 'DIVERGED'):
            done.append((run, mk))
        elif run_distance(run) in paused_distances():
            paused.append(run)
        elif mk == 'FAILED':
            failed.append(run)
        elif out:
            running.append((run, out))
        else:
            queued.append(run)
    md = ['# Depth vs distance: status', '', f'Updated {now:%Y-%m-%d %H:%M} UTC by dvd.py report.', '']
    for d in DISTANCES:
        f = os.path.join(HERE, 'lr', f'd{d}.json')
        if os.path.exists(f):
            j = json.load(open(f))
            md.append(f"- d{d} peak LR: {j.get('lr')} ({j['reason']})")
        f = os.path.join(HERE, 'lmin', f'd{d}.json')
        if os.path.exists(f):
            j = json.load(open(f))
            md.append(f"- d{d} L_min: {j.get('lmin')} ({j['reason']}); second seeds: {j['seed_runs']}")
    md += ['', '## Eval block, GRU / MWPM', '', '| d | ' + ' | '.join(f'L={L}' for L in DEPTHS) + ' |',
           '|---|' + '---|' * len(DEPTHS)]
    for d in DISTANCES:
        cells = []
        for L in DEPTHS:
            s = scores.get(sweep_name(d, L))
            s1 = scores.get(sweep_name(d, L, 1))
            c = f"{s['ratio']:.3f} (p {p_str(s['mc']['p_exact'])})" if s else ''
            if s1:
                c += f"; seed 1 {s1['ratio']:.3f}"
            cells.append(c)
        md.append(f'| {d} | ' + ' | '.join(cells) + ' |')
    left = {run: eta_hours(run, out) for run, out in running}
    left.update({run: eta_hours(run, None) for run in queued})
    n_par = max(1, len(running))
    md += ['', f'## Done ({len(done)})', ''] + [f'- {r} ({m})' for r, m in done]
    md += ['', f'## Running ({len(running)})', '']
    for run, out in running:
        ep = epochs_of(out)
        md.append(f'- {run}: epoch {max(ep) if ep else 0}, about {left[run]:.1f} h left')
    if failed:
        md += ['', f'## Failed ({len(failed)}), retried when queue.sh runs again', ''] + [f'- {r}' for r in failed]
    md += ['', f'## Queued ({len(queued)})', ''] + [f'- {r} (about {left[r]:.1f} h)' for r in queued]
    if paused:
        md += ['', f'## Paused ({len(paused)}), distances in queue/paused_d; resume from the last snapshot', '']
        md += [f'- {r}' for r in paused]
    tot = sum(left.values())
    md += ['', f'Remaining run-hours about {tot:.0f}; at the current {n_par} concurrent runs the queue needs '
           f'roughly {tot / n_par:.0f} h more (ETA about {(now + datetime.timedelta(hours=tot / n_par)):%Y-%m-%d %H:%M} UTC). '
           'Second-seed runs are not counted until L_min is known; paused runs are not counted.', '']
    open(os.path.join(HERE, 'STATUS.md'), 'w').write('\n'.join(md))
    print(f'wrote {path} ({len(rows)} rows) and STATUS.md')


def plan_table():
    lines = ['| run | d | L | width | parameters | peak LR | epochs | est. hours |', '|---|---|---|---|---|---|---|---|']
    total = {}
    for d in DISTANCES:
        sub = 0
        for run in scan_names(d):
            r = parse(run)
            h = est_hours(d, r['L'], SCAN_EPOCHS, False)
            sub += h
            lines.append(f"| {run} | {d} | {r['L']} | u{width(d)} | {n_params(d, r['L']):,} | {r['lr']} "
                         f"| {SCAN_EPOCHS} of {EPOCHS[d]} | {h:.1f} |")
        for L in DEPTHS:
            h = est_hours(d, L, EPOCHS[d], True)
            sub += h
            lines.append(f"| {sweep_name(d, L)} | {d} | {L} | u{width(d)} | {n_params(d, L):,} | from scan "
                         f"| {EPOCHS[d]} | {h:.1f} |")
        seeds = 2 * est_hours(d, 3, EPOCHS[d], True)
        lines.append(f"| second seeds at L_min - 1 and L_min | {d} | ? | u{width(d)} | | from scan | {EPOCHS[d]} "
                     f"| {seeds:.1f} (two runs, priced at L = 3) |")
        total[d] = sub + seeds
    lines += ['', '| d | est. run-hours |', '|---|---|'] + [f'| {d} | {total[d]:.0f} |' for d in DISTANCES]
    lines.append(f'| all | {sum(total.values()):.0f} |')
    print('\n'.join(lines))


if __name__ == '__main__':
    cmd, *rest = sys.argv[1:] or ['']
    if cmd == 'info':
        info(rest[0])
    elif cmd == 'kill-check':
        sys.exit(kill_check(rest[0]))
    elif cmd == 'lr-rule':
        lr_rule(int(rest[0]))
    elif cmd == 'lmin':
        lmin(int(rest[0]))
    elif cmd == 'report':
        report()
    elif cmd == 'plan-table':
        plan_table()
    elif cmd == 'scan-names':
        print(' '.join(scan_names(int(rest[0]))))
    elif cmd == 'sweep-name':
        print(sweep_name(int(rest[0]), int(rest[1])))
    elif cmd == 'finished':
        sys.exit(0 if finished(rest[0]) else 1)
    else:
        raise SystemExit(__doc__)
