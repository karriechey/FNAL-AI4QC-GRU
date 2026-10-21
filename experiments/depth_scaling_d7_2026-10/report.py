# Created: 2026-10-21
# Last modified: 2026-10-21

"""Depth-scaling results table and decision rules (prereg/depth_scaling_d7_2026-10.md).

~/tfenv/bin/python report.py   -> <repo>/GRU_depth_scaling_runs_<date>.csv, depth_sweep/report.md
McNemar b = this run right and the other decoder wrong.
"""
import argparse
import csv
import datetime
import glob
import json
import os
import re
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'src'))
from helpers import mcnemar_from_correct
from pair_two_decoders import load_dump

N_DET = 336
MARGIN = 0.03                # resolvable gap, from seed spread
P_SIG = 0.01
LOSS_TOL = 0.01              # final train loss "higher" means more than 1% higher
MATCH = {'match_d7_L04_u172': 4, 'match_d7_L08_u255': 8}

ap = argparse.ArgumentParser()
ap.add_argument('--kd-runs', default=os.path.expanduser(
    '~/FNAL-AI4QC-GRU-kd/experiments/kd_pilot_p004_2026-10/runs'))
ap.add_argument('--date', default=datetime.date.today().isoformat())
args = ap.parse_args()


def newest(pattern):
    found = sorted(glob.glob(pattern))
    return found[-1] if found else None


def p_str(p):
    return '<1e-300' if p == 0 else f'{p:.2g}'


def run_log(out):
    f = os.path.join(out, 'seed0', 'run.log')
    return open(f, errors='replace').read() if os.path.exists(f) else ''


def sec_per_epoch(out):
    s = [int(x) for x in re.findall(r'- (\d+)s/epoch', run_log(out))]
    return round(statistics.median(s[1:]), 0) if len(s) > 1 else ''


def runtime_h(out):
    t0 = datetime.datetime.strptime(re.search(r'_(\d{8}T\d{6}Z)$', out).group(1), '%Y%m%dT%H%M%SZ')
    t0 = t0.replace(tzinfo=datetime.timezone.utc).timestamp()
    t1 = max(os.path.getmtime(f) for f in glob.glob(os.path.join(out, '**'), recursive=True) if os.path.isfile(f))
    return round((t1 - t0) / 3600, 2)


def final_loss(out):
    h = glob.glob(os.path.join(out, 'seed0', 'runs', '*.history.json'))
    if h:
        return json.load(open(h[0]))['loss'][-1]
    v = re.findall(r'^\d+/\d+ - \d+s - loss: (\S+)', run_log(out), re.M)
    return float(v[-1]) if v else None


def env_of(out):
    return dict(re.findall(r'^(\w+)=(.*)$', open(os.path.join(out, 'env.txt')).read(), re.M))


def mc(a, b):
    # (a vs b) on shared shots; b = a right, b wrong
    ia, ca, _, ta = a
    ib, cb, _, tb = b
    shared, pa, pb = np.intersect1d(ia, ib, assume_unique=True, return_indices=True)
    if len(shared) != len(ia) or len(shared) != len(ib) or not np.array_equal(ta[pa], tb[pb]):
        raise SystemExit('eval dumps do not cover the same shots')
    return mcnemar_from_correct(ca[pa], cb[pb])


def mc_str(m):
    return f"b={m['rcnn_only']} c={m['mwpm_only']} n_disc={m['n_discordant']} p={p_str(m['p_exact'])}"


rows, dumps, ratio, loss = [], {}, {}, {}

# Part A
cls = os.path.join(HERE, 'lr_scan', 'classification.csv')
for r in (csv.DictReader(open(cls)) if os.path.exists(cls) else []):
    out = os.path.join(HERE, 'lr_scan', r['folder'])
    rows.append({'Run': r['run'], 'Part': 'A', 'L': r['L'], 'Width': 'u70',
                 'Parameters': 25061 + 29610 * (int(r['L']) - 1), 'Peak LR': r['peak_lr'],
                 'eta x N_det': r['eta_ndet'], 'Status': r['status'], 'Killed epoch': r['killed_epoch'],
                 'Final train loss': round(float(r['final_train_loss']), 5) if r['final_train_loss'] not in ('', 'nan') else r['final_train_loss'],
                 's/epoch': sec_per_epoch(out), 'runtime (h)': runtime_h(out), 'Date': args.date,
                 'Notes': f"{r['epochs']} epochs of the 80-epoch schedule; longest rise {r['longest_rise_epochs']} epochs, "
                          f"max loss / running min {r['max_loss_over_running_min']}" + (f"; {r['note']}" if r['note'] else '')})

# Part B
sweep_lr = open(os.path.join(HERE, 'depth_sweep', 'sweep_lr.txt')).read().strip() \
    if os.path.exists(os.path.join(HERE, 'depth_sweep', 'sweep_lr.txt')) else None
names = [f'depth_d7_L{L:02d}_u70' for L in range(1, 11)] + list(MATCH)
sel_csv = os.path.join(HERE, 'depth_sweep', 'selection_scores.csv')
sel_rows = list(csv.DictReader(open(sel_csv))) if os.path.exists(sel_csv) else []
for name in names:
    out = newest(os.path.join(HERE, 'depth_sweep', f'{name}_2*'))
    srows, notes = sel_rows, []
    if out is None and name == 'depth_d7_L02_u70' and sweep_lr == '1e-3':
        out = newest(os.path.join(args.kd_runs, 'hl7_2x70_2*'))
        kd_sel = os.path.join(args.kd_runs, 'selection_scores.csv')
        srows = [dict(r, model=name) for r in csv.DictReader(open(kd_sel)) if r['model'] == 'hl7_2x70']
        notes.append(f'kd pilot hl7_2x70 ({os.path.basename(out)}), same recipe and shots; epochs saved from 40, '
                     'so no epoch-20 score; ran alongside the d7 kd pilot runs, so s/epoch is not comparable')
    if out is None:
        continue
    env = env_of(out)
    L = len(env['UNITS'].split())
    width = int(env['UNITS'].split()[0])
    lr = env['LR']
    killed = [open(os.path.join(out, f)).read().strip() for f in ('KILLED', 'DIVERGED') if os.path.exists(os.path.join(out, f))]
    done = os.path.exists(os.path.join(out, 'DONE'))
    status = 'Done' if done else ('diverged' if killed else 'running or failed')
    by_ep = {int(r['epoch']): float(r['ratio']) for r in srows if r['model'] == name}
    row = {'Run': name, 'Part': 'B', 'L': L if width == 70 else MATCH[name], 'Width': f'u{width}',
           'Parameters': 25061 + 29610 * (L - 1) if width == 70 else {172: 114209, 255: 232816}[width],
           'Peak LR': lr, 'eta x N_det': round(float(lr) * N_DET, 2), 'Status': status,
           'Killed epoch': re.search(r'(?:ep|epoch) (\d+)', killed[0]).group(1) if killed else '',
           'Final train loss': round(final_loss(out), 5) if final_loss(out) is not None else '',
           's/epoch': '' if 'hl7_2x70' in out else sec_per_epoch(out),
           'runtime (h)': runtime_h(out), 'Date': args.date}
    for e in (20, 40, 60, 80):
        row[f'Sel x MWPM ep{e}'] = by_ep.get(e, '')
    if killed:
        notes.append(killed[0])
    if done:
        sel = json.load(open(os.path.join(out, 'selected.json')))
        dump = os.path.join(out, 'eval_selected_per_shot.npz')
        d = load_dump(dump)
        m_ = load_dump(dump, 'mwpm_correct')
        dumps[name] = d
        err, m_err = int((~d[1]).sum()), int((~m_[1]).sum())
        ratio[name] = err / m_err
        loss[name] = final_loss(out)
        row.update({'Selected epoch': sel['epoch'], 'p_L': round(err / len(d[1]), 6),
                    'MWPM': round(m_err / len(d[1]), 6), 'Relative to MWPM': round(err / m_err, 4),
                    'McNemar vs MWPM': mc_str(mc(d, m_)),
                    'Last-20 slope': sel['last20_slope_per_epoch'], 'Last-20 t': sel['last20_slope_t'],
                    'Converged': '' if sel['not_converged'] is None else ('no' if sel['not_converged'] else 'yes')})
    row['Notes'] = '; '.join(notes)
    rows.append(row)

# paired comparisons on the eval block
by_name = {r['Run']: r for r in rows}
comp = {}
for L in range(2, 11):
    a, b = f'depth_d7_L{L:02d}_u70', f'depth_d7_L{L - 1:02d}_u70'
    if a in dumps and b in dumps:
        comp[a] = (b, mc(dumps[a], dumps[b]))
for m, L in MATCH.items():
    s = f'depth_d7_L{L:02d}_u70'
    if m in dumps and s in dumps:
        comp[m] = (s, mc(dumps[m], dumps[s]))
for a, (b, m) in comp.items():
    by_name[a]['McNemar vs comparison'] = f'vs {b}: ' + mc_str(m)

cols = ['Run', 'Part', 'L', 'Width', 'Parameters', 'Peak LR', 'eta x N_det', 'Status', 'Killed epoch',
        'Sel x MWPM ep20', 'Sel x MWPM ep40', 'Sel x MWPM ep60', 'Sel x MWPM ep80', 'Selected epoch',
        'p_L', 'MWPM', 'Relative to MWPM', 'McNemar vs MWPM', 'McNemar vs comparison', 'Final train loss',
        'Last-20 slope', 'Last-20 t', 'Converged', 's/epoch', 'runtime (h)', 'Date', 'Notes']
path = os.path.join(REPO, f'GRU_depth_scaling_runs_{args.date}.csv')
with open(path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, '') for c in cols})
print(f'wrote {path} ({len(rows)} rows)')

# decision rules
md = ['# Depth scaling d=7: decision rules', '', f'Sweep LR: {sweep_lr}', '']
dep = {L: f'depth_d7_L{L:02d}_u70' for L in range(1, 11)}
if all(dep[L] in ratio for L in range(1, 11)):
    l_star = 10
    for L in range(1, 10):
        gain = (ratio[dep[L]] - ratio[dep[L + 1]]) / ratio[dep[L]]
        p = comp[dep[L + 1]][1]['p_exact']
        md.append(f'- L={L} -> {L + 1}: ratio {ratio[dep[L]]:.4f} -> {ratio[dep[L + 1]]:.4f} '
                  f'(gain {gain:+.1%}), McNemar p {p_str(p)}')
        if l_star == 10 and (gain < MARGIN or p >= P_SIG):
            l_star = L
    md += ['', f'L* = {l_star}' + (' (no saturation by L=10)' if l_star == 10 else ''), '']
    for L in range(2, 11):
        a, b = dep[L], dep[L - 1]
        m = comp[a][1]
        if ratio[a] >= (1 + MARGIN) * ratio[b] and m['p_exact'] < P_SIG:
            kind = ('optimization failure' if loss[a] > (1 + LOSS_TOL) * loss[b] else 'unused capacity')
            md.append(f'- degradation at L={L}: ratio {ratio[a]:.4f} vs {ratio[b]:.4f} at L={L - 1}; final train loss '
                      f'{loss[a]:.5f} vs {loss[b]:.5f}: {kind}')
else:
    md.append(f'L* not evaluated: eval scores missing for L = '
              f'{[L for L in range(1, 11) if dep[L] not in ratio]}')
md.append('')
for m, L in MATCH.items():
    s = dep[L]
    if m in ratio and s in ratio:
        p = comp[m][1]['p_exact']
        if ratio[s] <= (1 - MARGIN) * ratio[m] and p < P_SIG:
            v = 'stack wins'
        elif ratio[m] <= (1 - MARGIN) * ratio[s] and p < P_SIG:
            v = 'single layer wins'
        else:
            v = 'unresolved'
        md.append(f'- {m} ({ratio[m]:.4f}) vs L={L} stack ({ratio[s]:.4f}): McNemar p {p_str(p)}: {v}')
md.append('')
open(os.path.join(HERE, 'depth_sweep', 'report.md'), 'w').write('\n'.join(md))
print('\n'.join(md))
