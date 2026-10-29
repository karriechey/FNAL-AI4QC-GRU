# Created: 2026-10-29
# Last modified: 2026-10-29
# X-basis teachers: GRU_xbasis_runs.csv (one row per run) and zx_comparison.{csv,md}, Z vs X per d.
# Z and X are different shots, so the comparison is unpaired: GRU / MWPM with a 95% interval from
# a multinomial bootstrap of each run's paired 2x2 table, and the X - Z difference of the ratios.
#   ~/tfenv/bin/python report_x.py
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
sys.path.insert(0, os.path.join(REPO, 'experiments', 'depth_vs_distance'))
from helpers import mcnemar_from_correct
from pair_two_decoders import load_dump
import dvd

N_EVAL = 1_800_000
BOOT = 20000
RUNS = {  # X run: d, units, peak LR, epochs, Z teacher
    'x5_u280': (5, (280,), '3e-3, no warmup', 80, 'f5w_p004'),
    'x7_2xu280': (7, (280, 280), '1e-3, warmup 3', 80, 'f7w_p004'),
    'x9_3xu280': (9, (280, 280, 280), '5e-4, warmup 3', 240, 'f3280e240'),
}
# Z teachers on the Z FORMAL eval block [15.2M, 17M), from GRU_error_rate_runs_2026-10-20.csv:
# GRU p_L, MWPM p_L, McNemar b (GRU right, MWPM wrong), c (MWPM right, GRU wrong), selected epoch
Z = {'f5w_p004': (0.004923, 0.007571, 7280, 2515, 80),
     'f7w_p004': (0.00271, 0.004291, 4846, 2000, 80),
     'f3280e240': (0.001413, 0.002267, 2809, 1271, 240)}
PARAMS = {5: {1: 256481}, 7: {2: 747881}, 9: {3: 1246001}}


def cells(n, e_gru, e_mwpm, b, c):
    # (both wrong, GRU only right, MWPM only right, both right) from the paired counts
    w = round(((e_gru - c) + (e_mwpm - b)) / 2)
    return np.array([w, b, c, n - w - b - c], dtype=float)


def boot_ratios(cnt, rng):
    draws = rng.multinomial(int(cnt.sum()), cnt / cnt.sum(), size=BOOT)
    return (draws[:, 0] + draws[:, 2]) / (draws[:, 0] + draws[:, 1])


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return f'[{lo:.4f}, {hi:.4f}]'


def p_str(p):
    return '<1e-300' if p == 0 else f'{p:.2g}'


def main():
    rng = np.random.default_rng(0)
    rows, comp = [], []
    for run, (d, units, lr, epochs, zrun) in RUNS.items():
        found = sorted(glob.glob(os.path.join(HERE, 'runs', f'{run}_2*')))
        out = found[-1] if found else None
        L = len(units)
        row = {'Run': run, 'Basis': 'X', 'd': d, 'Width': ' x '.join(f'u{u}' for u in units) if L > 1 else f'u{units[0]}',
               'Parameters': PARAMS[d][L], 'Z teacher (same recipe)': zrun,
               'LR schedule': f'cosine to 3e-5 over {epochs} ep, peak {lr}', 'Max epoch': epochs,
               'Data': 'fresh 100M/epoch, X basis', 'Date': datetime.date.today().isoformat()}
        if out is None:
            row['Status'] = 'not started'
            rows.append(row)
            continue
        mk = dvd.marker(out, 'DONE', 'KILLED', 'DIVERGED', 'FAILED')
        row['Status'] = {'DONE': 'Done', 'KILLED': 'killed', 'DIVERGED': 'diverged', 'FAILED': 'failed'}.get(mk, 'running')
        ep = dvd.epochs_of(out)
        secs = [ep[e]['s'] for e in sorted(ep)][1:]
        row['s/epoch'] = round(statistics.median(secs)) if secs else ''
        row['runtime (h)'] = dvd.runtime_h(out)
        row['Final train loss'] = round(ep[max(ep)]['loss'], 5) if ep else ''
        notes = []
        for f in ('KILLED', 'DIVERGED'):
            if os.path.exists(os.path.join(out, f)):
                notes.append(open(os.path.join(out, f)).read().strip())
        if mk == 'DONE':
            sel = json.load(open(os.path.join(out, 'selected.json')))
            dump = os.path.join(out, 'eval_selected_per_shot.npz')
            g, m = load_dump(dump), load_dump(dump, 'mwpm_correct')
            if not (len(g[0]) == len(m[0]) == N_EVAL and (g[0] == m[0]).all()):
                raise SystemExit(f'{run}: GRU and MWPM dumps do not cover the same {N_EVAL:,} eval shots')
            mc = mcnemar_from_correct(g[1], m[1])
            e_g, e_m = int((~g[1]).sum()), int((~m[1]).sum())
            by_ep = {int(r['epoch']): float(r['ratio']) for r in csv.DictReader(open(os.path.join(out, 'selection_scores.csv')))}
            row.update({'Selected epoch': sel['epoch'], 'p_L': round(e_g / N_EVAL, 6), 'MWPM': round(e_m / N_EVAL, 6),
                        'Relative to MWPM': round(e_g / e_m, 4),
                        'McNemar vs MWPM': f"b={mc['rcnn_only']} c={mc['mwpm_only']} n_disc={mc['n_discordant']} p={p_str(mc['p_exact'])}",
                        'Selection-block x MWPM': by_ep.get(sel['epoch'], ''),
                        'Last-20 slope': sel['last20_slope_per_epoch'], 'Last-20 t': sel['last20_slope_t'],
                        'Converged': '' if sel['not_converged'] is None else ('no' if sel['not_converged'] else 'yes')})
            if sel['not_converged']:
                notes.append('still improving: last-20 selection slope negative with t < -2')
            zg, zm, zb, zc, zsel = Z[zrun]
            zcells = cells(N_EVAL, round(zg * N_EVAL), round(zm * N_EVAL), zb, zc)
            xcells = np.array([int((~g[1] & ~m[1]).sum()), mc['rcnn_only'], mc['mwpm_only'],
                               int((g[1] & m[1]).sum())], dtype=float)
            zr, xr = boot_ratios(zcells, rng), boot_ratios(xcells, rng)
            comp.append({'d': d, 'model': row['Width'], 'Z run': zrun, 'Z GRU p_L': zg, 'Z MWPM p_L': zm,
                         'Z GRU / MWPM': round(zg / zm, 4), 'Z 95% CI': ci(zr), 'Z selected epoch': zsel,
                         'X run': run, 'X GRU p_L': row['p_L'], 'X MWPM p_L': row['MWPM'],
                         'X GRU / MWPM': row['Relative to MWPM'], 'X 95% CI': ci(xr), 'X selected epoch': sel['epoch'],
                         'X - Z ratio': round(row['Relative to MWPM'] - zg / zm, 4), 'X - Z 95% CI': ci(xr - zr),
                         'MWPM X / Z p_L': round(row['MWPM'] / zm, 3)})
        row['Notes'] = '; '.join(notes)
        rows.append(row)

    cols = ['Run', 'Basis', 'd', 'Width', 'Parameters', 'Z teacher (same recipe)', 'LR schedule', 'Data', 'Max epoch',
            'Status', 'Selected epoch', 'p_L', 'MWPM', 'Relative to MWPM', 'McNemar vs MWPM', 'Selection-block x MWPM',
            'Final train loss', 'Last-20 slope', 'Last-20 t', 'Converged', 's/epoch', 'runtime (h)', 'Date', 'Notes']
    with open(os.path.join(HERE, 'GRU_xbasis_runs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})
    if comp:
        with open(os.path.join(HERE, 'zx_comparison.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(comp[0]))
            w.writeheader()
            w.writerows(comp)
        md = ['| d | model | Z GRU / MWPM | X GRU / MWPM | X - Z [95% CI] | MWPM p_L Z / X |', '|---|---|---|---|---|---|']
        for c in comp:
            md.append(f"| {c['d']} | {c['model']} | {c['Z GRU / MWPM']} {c['Z 95% CI']} | {c['X GRU / MWPM']} {c['X 95% CI']} "
                      f"| {c['X - Z ratio']:+.4f} {c['X - Z 95% CI']} | {c['Z MWPM p_L']} / {c['X MWPM p_L']} |")
        open(os.path.join(HERE, 'zx_comparison.md'), 'w').write('\n'.join(md) + '\n')
        print('\n'.join(md))
    print(f'{len(rows)} rows -> GRU_xbasis_runs.csv')


if __name__ == '__main__':
    main()
