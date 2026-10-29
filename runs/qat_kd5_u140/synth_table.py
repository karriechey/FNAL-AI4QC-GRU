# Created: 2026-10-29 | Last updated: 2026-10-29
# EAF Vivado HLS csynth results + eval-block pL -> results/synth_rf70.csv, results/synth_rf_sweep.csv
# and figures/pareto_pL_vs_lut.png
#   python3 synth_table.py
import csv
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
SYNTH_DIR = os.path.join(RES, 'synth_vivado_rf70_2026-10-29')
RF_SWEEP_DIR = os.path.join(RES, 'synth_vivado_rf_sweep_2026-10-29')
CLOCK_NS = 5.0

points = {r['config']: r for r in csv.DictReader(open(os.path.join(RES, 'all_points.csv')))}


def synth_row(s, source_dir):
    name = s['config'].removesuffix('_vivado')
    rf_match = re.search(r'_rf(\d+)$', name)
    reuse_factor = int(rf_match.group(1)) if rf_match else 70
    config = re.sub(r'_rf\d+$', '', name)
    p = points[config]
    row = {'config': config, 'weight_bits': int(p['weight_bits']), 'state_bits': int(p['state_bits']),
           'reuse_factor': reuse_factor, 'hls_tool': 'Vivado HLS 2020.1', 'part': s['part'],
           'target_clock_ns': float(s['target_clock_ns']), 'est_clock_ns': float(s['est_clock_ns']),
           'latency_best_cycles': int(s['latency_best_cycles']), 'latency_worst_cycles': int(s['latency_worst_cycles']),
           'latency_worst_us': int(s['latency_worst_cycles']) * CLOCK_NS / 1000,
           'ii_cycles': int(s['ii_max']), 'ii_us': int(s['ii_max']) * CLOCK_NS / 1000}
    for res in ('LUT', 'FF', 'DSP', 'BRAM_18K', 'URAM'):
        row[res] = int(s[res])
        row[f'{res}_pct'] = round(100 * int(s[res]) / int(s[f'available_{res}']), 2)
    row['pL'] = float(p['pL'])
    row['pL_over_fp32'] = float(p['pL_over_fp32'])
    row['pL_over_mwpm'] = float(p['pL_over_mwpm'])
    row['hls4ml_pL_1M'] = float(p['hls4ml_pL'])
    row['source'] = os.path.relpath(source_dir, HERE)
    return row


rows = [synth_row(s, SYNTH_DIR) for s in csv.DictReader(open(os.path.join(SYNTH_DIR, 'synth_all.csv')))]
rows.sort(key=lambda r: r['LUT'])

rf_rows = [r for r in rows if r['config'] == 'qat_w12_s6_seed0']
for f in glob.glob(os.path.join(RF_SWEEP_DIR, '*', 'synth_results.csv')):
    rf_rows += [synth_row(s, os.path.dirname(f)) for s in csv.DictReader(open(f))]
rf_rows.sort(key=lambda r: r['reuse_factor'])
with open(os.path.join(RES, 'synth_rf_sweep.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rf_rows[0]))
    w.writeheader()
    w.writerows(rf_rows)

with open(os.path.join(RES, 'synth_rf70.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

fp32 = float(points['qat_w12_s6_seed0']['pL_fp32'])
fig, ax = plt.subplots(figsize=(5.5, 4))
for r in rows:
    ax.scatter(r['LUT_pct'], r['pL'], s=30, color='C3' if r['config'] == 'qat_w12_s6_seed0' else 'C0')
    ax.annotate(f"w{r['weight_bits']} s{r['state_bits']}\nDSP {r['DSP']}", (r['LUT_pct'], r['pL']),
                textcoords='offset points', xytext=(6, -4), fontsize=7)
ax.axhline(fp32, color='k', lw=1, label='FP32 parent')
ax.axhline(1.05 * fp32, color='k', lw=0.8, ls=':', label='1.05 x FP32')
ax.set_xlabel('LUT (% of xcvu13p), RF 70, Vivado HLS 2020.1')
ax.set_ylabel('pL, eval block (1.8M shots)')
ax.set_xlim(29, 37)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'figures', 'pareto_pL_vs_lut.png'), dpi=150)

for r in rows + [r for r in rf_rows if r['reuse_factor'] != 70]:
    print(f"{r['config']} RF {r['reuse_factor']}: LUT {r['LUT']} ({r['LUT_pct']}%), FF {r['FF']} ({r['FF_pct']}%), DSP {r['DSP']} "
          f"({r['DSP_pct']}%), BRAM {r['BRAM_18K']} ({r['BRAM_18K_pct']}%), {r['latency_worst_us']} us, "
          f"pL {r['pL_over_fp32']:.3f}x FP32")
