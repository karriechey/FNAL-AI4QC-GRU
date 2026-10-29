# Created: 2026-10-28 | Last updated: 2026-10-28
# Collects PTQ, QAT and hls4ml results into results/all_points.csv and draws the per-axis plots
# and the Pareto plot (eval-block p_L vs total bits; no synthesis numbers on grace1).
#   bash hpy.sh summarize.py
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcommon as Q

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = os.path.join(Q.HERE, 'results')
FIG = os.path.join(Q.HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

rows = []
for r in csv.DictReader(open(os.path.join(RES, 'p1_ptq.csv'))):
    rows.append({'config': f"ptq_w{r['weight_bits']}_s{r['state_bits']}", 'method': 'PTQ (Phase 1 integer bits)',
                 'weight_bits': int(r['weight_bits']), 'state_bits': int(r['state_bits']),
                 'ler': float(r['ler']), 'ler_fp32': float(r['ler_fp32']), 'ler_mwpm': float(r['ler_mwpm']),
                 'b': int(r['b']), 'c': int(r['c']), 'n_disc': int(r['n_disc']), 'p_exact': float(r['p_exact']),
                 'status': '', 'best_epoch': ''})


def row_from_score(config, method, wb, sb, sc, status='', epoch=''):
    mc = sc['mcnemar_vs_fp32']
    return {'config': config, 'method': method, 'weight_bits': wb, 'state_bits': sb, 'ler': sc['ler'],
            'ler_fp32': sc['ler_fp32'], 'ler_mwpm': sc['ler_mwpm'], 'b': mc['b'], 'c': mc['c'],
            'n_disc': mc['n_disc'], 'p_exact': mc['p_exact'], 'status': status, 'best_epoch': epoch}


for res in sorted(glob.glob(os.path.join(Q.HERE, 'qat_runs', '*', 'result.json'))):
    r = json.load(open(res))
    name = r['tag']
    wb, sb = r['spec']['gru_weight'][0], r['spec']['gru_state'][0]
    rows.append(row_from_score(f'ptq_qatbits_{name}', 'PTQ (QAT integer bits)', wb, sb, r['start_score']))
    method = 'QAT' if '_lr' not in name else 'QAT diagnostic (LR 3e-5, not used for selection)'
    rows.append(row_from_score(f'qat_{name}', method, wb, sb, r['score'], r['status'], r['best_epoch']))

hls = {}
for res in sorted(glob.glob(os.path.join(RES, 'p3_*.json'))):
    r = json.load(open(res))
    hls[r['config'].replace(':', '_')] = r
for row in rows:
    h = hls.get(row['config'])
    row['hls4ml_agreement'] = h['agreement_vs_quantized_keras'] if h else ''
    row['hls4ml_ler'] = h['hls']['ler'] if h else ''
    row['hls4ml_n'] = h['n'] if h else ''
    row['total_bits'] = row['weight_bits'] + row['state_bits']
    row['ler_over_fp32'] = row['ler'] / row['ler_fp32']
    row['ler_over_mwpm'] = row['ler'] / row['ler_mwpm']
    row['acceptable'] = Q.acceptable(row) and not row['method'].startswith('QAT diagnostic')

with open(os.path.join(RES, 'all_points.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

fp32, mwpm = rows[0]['ler_fp32'], rows[0]['ler_mwpm']
style = {'PTQ (Phase 1 integer bits)': ('o-', 'C0'), 'PTQ (QAT integer bits)': ('s--', 'C2'), 'QAT': ('D-', 'C3')}
for axis, key, fixed_key in (('weights', 'weight_bits', 'state_bits'), ('state', 'state_bits', 'weight_bits')):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for method, (st, col) in style.items():
        one_seed = lambda r: r['method'].startswith('PTQ (Phase') or r['config'].endswith('_seed0')
        pts = sorted({r[key]: r for r in rows if r['method'] == method and r[fixed_key] == 16
                      and r[key] < 32 and one_seed(r)}.items())
        if pts:
            ax.plot([p[0] for p in pts], [p[1]['ler'] for p in pts], st, color=col, label=method)
    ax.axhline(fp32, color='k', lw=1, label=f'FP32 parent {fp32:.5f}')
    ax.axhline(1.05 * fp32, color='k', lw=0.8, ls=':', label='1.05 x FP32')
    ax.axhline(mwpm, color='C1', lw=1, label=f'MWPM {mwpm:.5f}')
    ax.set_yscale('log')
    ax.set_xlabel(f'{axis} bits (other group at 16, accumulators at {Q.ACCUM_BITS})')
    ax.set_ylabel('logical error rate, eval block (1.8M shots)')
    ax.invert_xaxis()
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f'ler_vs_{axis}_bits.png'), dpi=150)

fig, ax = plt.subplots(figsize=(5.5, 4))
for method, (st, col) in style.items():
    pts = [r for r in rows if r['method'] == method]
    ax.scatter([r['total_bits'] for r in pts], [r['ler'] for r in pts], marker=st[0], color=col, label=method, s=18)
ax.axhline(fp32, color='k', lw=1, label='FP32 parent')
ax.axhline(1.05 * fp32, color='k', lw=0.8, ls=':', label='1.05 x FP32')
ax.axhline(mwpm, color='C1', lw=1, label='MWPM')
ax.set_yscale('log')
ax.set_ylim(fp32 * 0.9, 0.06)
ax.set_xlabel('total bits (weight bits + state bits)')
ax.set_ylabel('logical error rate, eval block')
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'pareto_ler_vs_total_bits.png'), dpi=150)

ok = [r for r in rows if r['acceptable'] and r['method'] != 'PTQ (QAT integer bits)']
ok.sort(key=lambda r: (r['total_bits'], r['state_bits'], r['ler']))
print('acceptable, fewest total bits first:')
for r in ok[:8]:
    print(f"  {r['config']}: W{r['weight_bits']} S{r['state_bits']} p_L {r['ler']:.6f} ({r['ler_over_fp32']:.3f}x FP32)")

seeds = {}
for r in rows:
    if r['method'] == 'QAT' and '_seed' in r['config']:
        seeds.setdefault(r['config'].rsplit('_seed', 1)[0], []).append(r)
with open(os.path.join(RES, 'qat_seeds.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['config', 'n_seeds', 'ler_per_seed', 'ler_mean', 'ler_over_fp32_mean', 'ler_over_mwpm_mean',
                'acceptable_per_seed', 'acceptable_mean'])
    for k, rs in sorted(seeds.items()):
        m = sum(r['ler'] for r in rs) / len(rs)
        acc_mean = m <= 1.05 * fp32 and m < mwpm
        w.writerow([k, len(rs), ';'.join(f"{r['ler']:.6f}" for r in rs), f'{m:.6f}', f'{m / fp32:.4f}',
                    f'{m / mwpm:.4f}', ';'.join(str(r['acceptable']) for r in rs), acc_mean])
        if len(rs) > 1:
            print(f'{k}: {len(rs)} seeds, mean p_L {m:.6f} ({m / fp32:.4f}x FP32), acceptable on mean: {acc_mean}')
