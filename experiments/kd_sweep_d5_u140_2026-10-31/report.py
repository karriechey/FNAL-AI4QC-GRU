# Created: 2026-10-31 | Last updated: 2026-10-31
# Sweep table from every finished cell of configs.tsv: selected checkpoint (selection block [0, 10M)),
# eval-block [15.2M, 17M) pL with Wilson 95% interval, ratio to uncorrelated MWPM and to the teacher on the
# same shots, McNemar against the original alpha=0 T=1 student (kd5_u140), decision agreement with the
# teacher. Ranked by eval-block pL. Every cell must have the same eval shots in the same order.
#   ~/tfenv/bin/python report.py [--correlated <per_shot_d5_n1800000.npz>]
import argparse
import csv
import glob
import json
import os

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(HERE, '..', 'kd_pilot_p004_2026-10')
ap = argparse.ArgumentParser()
ap.add_argument('--correlated', default=None, help='correlated-MWPM per-shot decisions on the same eval shots')
args = ap.parse_args()


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def mcnemar(this_correct, base_correct):
    # b = this run right and the baseline wrong
    b = int((this_correct & ~base_correct).sum())
    c = int((~this_correct & base_correct).sum())
    return b, c, float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0


cells = [l.rstrip('\n').split('\t') for l in open(os.path.join(HERE, 'configs.tsv')) if not l.startswith('#')]
teacher = np.load(os.path.join(PILOT, 'teacher_check', 'd5', 'teacher_eval_per_shot.npz'))
shots, truth = teacher['shot_idx'], teacher['truth'].astype(bool)
assert shots.min() >= 15_200_000 and shots.max() < 17_000_000
t_correct = teacher['student_correct']
t_amb = np.abs(teacher['student_logit']) < np.log(0.95 / 0.05)     # teacher p in (0.05, 0.95)
corr = None
if args.correlated:
    zc = np.load(args.correlated)
    assert np.array_equal(zc['shot_idx'], shots)
    corr = zc['mwpm_correlated_pred'].astype(bool) == truth

found = {}
for name, alpha, temp, source in cells:
    base = os.path.join(PILOT, 'runs') if source == 'pilot' else os.path.join(HERE, 'keep')
    dirs = sorted(d for d in glob.glob(os.path.join(base, f'{name}_2*'))
                  if os.path.exists(os.path.join(d, 'eval_selected_per_shot.npz')))
    if dirs:
        found[name] = (alpha, temp, source, dirs[-1])
assert 'kd5_u140' in found, 'baseline kd5_u140 missing'
z0 = np.load(os.path.join(found['kd5_u140'][3], 'eval_selected_per_shot.npz'))
base_correct = z0['student_correct']

rows = []
for name, (alpha, temp, source, d) in found.items():
    z = np.load(os.path.join(d, 'eval_selected_per_shot.npz'))
    assert np.array_equal(z['shot_idx'], shots) and np.array_equal(z['truth'].astype(bool), truth), name
    sel = json.load(open(os.path.join(d, 'selected.json')))
    sc = z['student_correct']
    n, err = len(sc), int((~sc).sum())
    mwpm_err = int((~z['mwpm_correct']).sum())
    lo, hi = wilson(err, n)
    b, c, p = mcnemar(sc, base_correct)
    pred, t_pred = z['student_pred'].astype(bool), teacher['student_pred'].astype(bool)
    r = {'run': name, 'alpha': float(alpha), 'T': temp, 'source': source, 'selected_epoch': sel['epoch'],
         'sel_pL': sel['sel_p_L'], 'sel_x_mwpm': sel['sel_ratio'], 'not_converged': sel.get('not_converged'),
         'eval_errors': err, 'eval_n': n, 'eval_pL': err / n, 'eval_pL_wilson_lo': lo, 'eval_pL_wilson_hi': hi,
         'eval_x_mwpm': err / mwpm_err, 'eval_x_teacher': err / int((~t_correct).sum()),
         'mcnemar_vs_kd5_u140_b': b, 'mcnemar_vs_kd5_u140_c': c, 'mcnemar_vs_kd5_u140_p': p,
         'agree_teacher_all': float((pred == t_pred).mean()), 'agree_teacher_ambiguous': float((pred == t_pred)[t_amb].mean()),
         'weights': os.path.realpath(sel['weights']) if source == 'pilot' else os.path.join(d, os.path.basename(sel['weights']))}
    if corr is not None:
        r['eval_x_mwpm_correlated'] = err / int((~corr).sum())
    rows.append(r)
rows.sort(key=lambda r: (r['eval_pL'], r['sel_pL']))
for i, r in enumerate(rows, 1):
    r['rank'] = i

os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
keys = ['rank'] + [k for k in rows[0] if k != 'rank']
with open(os.path.join(HERE, 'results', 'sweep_results.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
json.dump({'eval_shots': [int(shots.min()), int(shots.max()) + 1], 'n_cells_found': len(rows),
           'n_cells_planned': len(cells), 'teacher_eval_errors': int((~t_correct).sum()),
           'mwpm_eval_errors': int((~z0['mwpm_correct']).sum()), 'rows': rows},
          open(os.path.join(HERE, 'results', 'sweep_results.json'), 'w'), indent=1)

lines = ['| rank | alpha | T | epoch | sel pL | eval pL (95% CI) | x MWPM | x teacher | vs kd5_u140 b / c (p) |',
         '|---|---|---|---|---|---|---|---|---|']
for r in rows:
    lines.append(f"| {r['rank']} | {r['alpha']:g} | {r['T']} | {r['selected_epoch']} | {r['sel_pL']:.6f} | "
                 f"{r['eval_pL']:.6f} ({r['eval_pL_wilson_lo']:.6f}-{r['eval_pL_wilson_hi']:.6f}) | {r['eval_x_mwpm']:.4f} | "
                 f"{r['eval_x_teacher']:.3f} | {r['mcnemar_vs_kd5_u140_b']} / {r['mcnemar_vs_kd5_u140_c']} "
                 f"({r['mcnemar_vs_kd5_u140_p']:.2g}) |")
open(os.path.join(HERE, 'results', 'sweep_table.md'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(f'{len(rows)} of {len(cells)} cells')
