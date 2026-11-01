# Created: 2026-11-01 | Last updated: 2026-11-01
# Round-2 table: every finished student in keep/ plus the references (round-2 teacher, kd5_u140, pilot
# kd5_u70 / hl5_u70), on the eval block. McNemar against kd5_u140 and, per width, round-2 teacher arm (ta)
# against the direct f5w arm (b = ta right, f5w wrong).
#   ~/tfenv/bin/python report_r2.py
import csv
import glob
import json
import os
import re

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(HERE, '..', 'kd_pilot_p004_2026-10')
choice = json.load(open(os.path.join(HERE, 'choice.json')))
params = lambda u: 3 * u * u + 76 * u + 1          # 1 x GRU u (reset_after=False, 24 inputs) + Dense 1


def latest(pattern):
    dirs = sorted(d for d in glob.glob(pattern) if os.path.exists(os.path.join(d, 'eval_selected_per_shot.npz')))
    return dirs[-1] if dirs else None


runs = {}
for d in sorted(glob.glob(os.path.join(HERE, 'keep', 'r2_*_2*'))):
    m = re.match(r'r2_(ta|f5w)_u(\d+)_a([\d.]+)_T([\d.]+)_\d', os.path.basename(d))
    runs[(m.group(1), int(m.group(2)))] = d
refs = {'kd5_u140': latest(os.path.join(PILOT, 'runs', 'kd5_u140_2*')),
        'kd5_u70': latest(os.path.join(PILOT, 'runs', 'kd5_u70_2*')),
        'hl5_u70': latest(os.path.join(PILOT, 'runs', 'hl5_u70_2*')),
        'round2_teacher': os.path.dirname(choice['teacher_weights'])}
if choice['r2_alpha'] == '0.0' and choice['r2_T'] == '1.0' and ('f5w', 70) not in runs:
    runs[('f5w', 70)] = refs['kd5_u70']

load = lambda d: np.load(os.path.join(d, 'eval_selected_per_shot.npz'))
base = load(refs['kd5_u140'])
shots, truth = base['shot_idx'], base['truth']
mwpm_err = int((~base['mwpm_correct']).sum())


def mcnemar(a, b):
    nb, nc = int((a & ~b).sum()), int((~a & b).sum())
    return nb, nc, float(binomtest(nb, nb + nc, 0.5).pvalue) if nb + nc else 1.0


def row(name, d, units, arm):
    z = load(d)
    assert np.array_equal(z['shot_idx'], shots) and np.array_equal(z['truth'], truth), name
    sel = json.load(open(os.path.join(d, 'selected.json')))
    sc = z['student_correct']
    err = int((~sc).sum())
    b, c, p = mcnemar(sc, base['student_correct'])
    return {'run': name, 'arm': arm, 'units': units, 'params': params(units), 'selected_epoch': sel['epoch'],
            'sel_pL': sel['sel_p_L'], 'eval_errors': err, 'eval_pL': err / len(sc), 'eval_x_mwpm': err / mwpm_err,
            'beats_mwpm': err < mwpm_err, 'not_converged': sel.get('not_converged'),
            'vs_kd5_u140_b': b, 'vs_kd5_u140_c': c, 'vs_kd5_u140_p': p, 'dir': d, 'correct': sc}


rows = [row(os.path.basename(d).rsplit('_', 1)[0], d, u, arm) for (arm, u), d in sorted(runs.items())]
rows += [row(k, refs[k], u, 'reference') for k, u in (('round2_teacher', 140), ('kd5_u140', 140),
                                                    ('kd5_u70', 70), ('hl5_u70', 70))]
by = {(r['arm'], r['units']): r for r in rows}
for r in rows:
    other = by.get(('f5w', r['units'])) if r['arm'] == 'ta' else None
    if other is not None:
        r['ta_vs_f5w_b'], r['ta_vs_f5w_c'], r['ta_vs_f5w_p'] = mcnemar(r['correct'], other['correct'])
for r in rows:
    del r['correct']

os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
keys = []
for r in rows:
    keys += [k for k in r if k not in keys]
with open(os.path.join(HERE, 'results', 'r2_results.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
json.dump({'choice': choice, 'mwpm_eval_errors': mwpm_err, 'rows': rows},
          open(os.path.join(HERE, 'results', 'r2_results.json'), 'w'), indent=1)
lines = ['| run | arm | units | params | epoch | eval pL | x MWPM | vs kd5_u140 b / c (p) | ta vs f5w b / c (p) |',
         '|---|---|---|---|---|---|---|---|---|']
for r in rows:
    tv = f"{r['ta_vs_f5w_b']} / {r['ta_vs_f5w_c']} ({r['ta_vs_f5w_p']:.2g})" if 'ta_vs_f5w_b' in r else ''
    lines.append(f"| {r['run']} | {r['arm']} | {r['units']} | {r['params']:,} | {r['selected_epoch']} | "
                 f"{r['eval_pL']:.6f} | {r['eval_x_mwpm']:.4f} | {r['vs_kd5_u140_b']} / {r['vs_kd5_u140_c']} "
                 f"({r['vs_kd5_u140_p']:.2g}) | {tv} |")
open(os.path.join(HERE, 'results', 'r2_table.md'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
