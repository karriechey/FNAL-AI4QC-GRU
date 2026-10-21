# Created: 2026-10-18
# Last modified: 2026-10-18

"""Distillation pilot report; McNemar b = this run right, the other decoder wrong.

~/tfenv/bin/python report.py --d 5     -> runs/report_d5.csv, .md
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '../../src'))
from helpers import mcnemar_from_correct

NOISE = ('One initialisation seed per run. Same-recipe seeds of the d9 teacher differ by about '
         '0.02x MWPM on the eval block (f3280 0.652x, f3280s1 0.633x), so smaller differences '
         'between runs, KD vs hard label included, are within single-seed noise even when the '
         'paired McNemar p is small.')

ap = argparse.ArgumentParser()
ap.add_argument('--d', type=int, required=True)
ap.add_argument('--smoke', action='store_true')
ap.add_argument('--band', type=float, nargs=2, default=[0.05, 0.95])
args = ap.parse_args()
base = os.path.join(HERE, 'smoke' if args.smoke else 'runs')

t = dict(re.findall(r'^(\w+)=(\S+)$', subprocess.run(
    ['bash', '-c', f'D={args.d} source {HERE}/teachers.sh; set'], capture_output=True, text=True).stdout, re.M))
t_params, t_sel_err = int(t['T_PARAMS']), int(t['T_SEL_ERR'])
tdump = np.load(os.path.join(HERE, 'teacher_check', f'd{args.d}', 'teacher_eval_per_shot.npz'))
t_correct = tdump['student_correct'].astype(bool)
t_pred = tdump['student_pred'].astype(np.int8)
t_prob = 1 / (1 + np.exp(-tdump['student_logit'].astype(np.float64)))
truth = tdump['truth'].astype(np.int8)
n = len(truth)


def p_str(p):
    return '<1e-300' if p == 0 else f'{p:.2g}'


runs = {}
for done in sorted(glob.glob(os.path.join(base, f'*{args.d}_*', 'DONE'))):
    out = os.path.dirname(done)
    run = re.sub(r'_\d{8}T\d{6}Z$', '', os.path.basename(out))
    if not re.match(rf'^(kd|hl){args.d}_', run):
        continue
    sel = json.load(open(os.path.join(out, 'selected.json')))
    dump = np.load(os.path.join(out, 'eval_selected_per_shot.npz'))
    if not (np.array_equal(dump['shot_idx'], tdump['shot_idx']) and np.array_equal(dump['truth'], tdump['truth'])):
        raise SystemExit(f'{run}: eval shots differ from the teacher dump')
    ev = [r for r in csv.DictReader(open(os.path.join(base, 'eval_selected.csv')))
          if os.path.abspath(r['weights_path']) == os.path.abspath(sel['weights'])][-1]
    runs[run] = dict(out=out, sel=sel, dump=dump, n_params=int(ev['n_params']))

rows = []
for run, r in runs.items():
    s, dmp = r['sel'], r['dump']
    corr = dmp['student_correct'].astype(bool)
    err = int((~corr).sum())
    m_corr = dmp['mwpm_correct'].astype(bool)
    vm = mcnemar_from_correct(corr, m_corr)
    vt = mcnemar_from_correct(corr, t_correct)
    pred = dmp['student_pred'].astype(np.int8)
    amb = (t_prob > args.band[0]) & (t_prob < args.band[1])
    ag = dict(agree_all=float((pred == t_pred).mean()), agree_ambiguous=float((pred[amb] == t_pred[amb]).mean()),
              n_ambiguous=int(amb.sum()))
    ctrl = 'hl' + run[2:]
    vh = mcnemar_from_correct(corr, runs[ctrl]['dump']['student_correct'].astype(bool)) \
        if run.startswith('kd') and ctrl in runs else None
    rows.append(dict(
        run=run, d=args.d, arm='distilled' if run.startswith('kd') else 'hard label',
        n_params=r['n_params'], compression=round(t_params / r['n_params'], 2), epoch=s['epoch'],
        sel_p_L=s['sel_p_L'], sel_ratio_mwpm=s['sel_ratio'], sel_ratio_teacher=round(s['sel_errors'] / t_sel_err, 4),
        eval_errors=err, eval_p_L=round(err / n, 6), eval_ratio_mwpm=round(err / int((~m_corr).sum()), 4),
        eval_ratio_teacher=round(err / int((~t_correct).sum()), 4),
        mwpm_b=vm['rcnn_only'], mwpm_c=vm['mwpm_only'], mwpm_n_disc=vm['n_discordant'], mwpm_p=vm['p_exact'],
        teacher_b=vt['rcnn_only'], teacher_c=vt['mwpm_only'], teacher_n_disc=vt['n_discordant'], teacher_p=vt['p_exact'],
        hl_b='' if vh is None else vh['rcnn_only'], hl_c='' if vh is None else vh['mwpm_only'],
        hl_n_disc='' if vh is None else vh['n_discordant'], hl_p='' if vh is None else vh['p_exact'],
        agree_all=round(ag['agree_all'], 5), agree_ambiguous=round(ag['agree_ambiguous'], 4),
        n_ambiguous=ag['n_ambiguous'],
        last20_slope=s['last20_slope_per_epoch'], last20_t=s['last20_slope_t'],
        converged='' if s['not_converged'] is None else ('no' if s['not_converged'] else 'yes')))

tag = f'report_d{args.d}'
with open(os.path.join(base, tag + '.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['run'])
    w.writeheader()
    w.writerows(rows)
md = [f'# Distillation pilot, d={args.d}, p=0.004 (teacher {t["T_NAME"]}, {t_params:,} parameters)', '',
      f'Eval block [15.2M, 17M), {n:,} shots; selection block [0, 10M). b = this run right and the other '
      f'decoder wrong. Agreement band: teacher p in ({args.band[0]}, {args.band[1]}).', '',
      '| run | params | compr. | epoch | sel x MWPM | sel x teacher | eval p_L | eval x MWPM | eval x teacher '
      '| vs MWPM b/c/n/p | vs teacher b/c/n/p | vs hard label b/c/n/p | agree all / band | converged |',
      '|' + '---|' * 14]
for r in rows:
    hl = '' if r['hl_b'] == '' else f"{r['hl_b']}/{r['hl_c']}/{r['hl_n_disc']}/{p_str(r['hl_p'])}"
    md.append(f"| {r['run']} | {r['n_params']:,} | {r['compression']}x | {r['epoch']} | {r['sel_ratio_mwpm']:.4f} "
              f"| {r['sel_ratio_teacher']:.4f} | {r['eval_p_L']:.6f} | {r['eval_ratio_mwpm']:.4f} "
              f"| {r['eval_ratio_teacher']:.4f} | {r['mwpm_b']}/{r['mwpm_c']}/{r['mwpm_n_disc']}/{p_str(r['mwpm_p'])} "
              f"| {r['teacher_b']}/{r['teacher_c']}/{r['teacher_n_disc']}/{p_str(r['teacher_p'])} | {hl} "
              f"| {r['agree_all']:.4f} / {r['agree_ambiguous']:.4f} | {r['converged']} |")
md += ['', NOISE, '']
open(os.path.join(base, tag + '.md'), 'w').write('\n'.join(md))
print('\n'.join(md))
