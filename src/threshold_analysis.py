# Created: 2026-09-26
# Last modified: 2026-09-26

"""LER table with 95% Wilson intervals, LER-vs-p plot, and the d5/d7 and d7/d9 threshold crossings.

Crossings interpolate linearly in (log p, log LER). Per-round LER
p_round = (1 - (1 - 2 LER)^(1/r)) / 2 is reported as a secondary number.
Writes ler_table.csv, threshold.png, crossings.json to --out.

python threshold_analysis.py <sweep_dir>/stats.csv --out <sweep_dir>
"""
import argparse
import csv
import json
import math
import os

import numpy as np
import sinter

FOWLER = 0.0057


def wilson(k, n, z=1.959964):
    if n == 0:
        return float('nan'), float('nan')
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def per_round(ler, r):
    return (1 - (1 - 2 * ler) ** (1.0 / r)) / 2


def crossing(ps, a, b):
    # first p where a (smaller d) crosses above b; None if no crossing in the grid
    la, lb, lp = np.log(a), np.log(b), np.log(ps)
    diff = la - lb
    for i in range(len(ps) - 1):
        if diff[i] < 0 <= diff[i + 1] or diff[i] > 0 >= diff[i + 1]:
            t = diff[i] / (diff[i] - diff[i + 1])
            return float(np.exp(lp[i] + t * (lp[i + 1] - lp[i])))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stats')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = {}
    for s in sinter.read_stats_from_csv_files(a.stats):
        m = s.json_metadata
        key = (m['d'], m['p'])
        k, n = s.errors, s.shots
        if key in rows:                      # merge resumed partial rows
            k += rows[key]['errors']; n += rows[key]['shots']
        rows[key] = dict(d=m['d'], p=m['p'], r=m['r'], shots=n, errors=k)
    table = []
    for (d, p), r in sorted(rows.items()):
        ler = r['errors'] / r['shots']
        lo, hi = wilson(r['errors'], r['shots'])
        table.append(dict(r, ler=ler, ci95_lo=lo, ci95_hi=hi,
                          ler_per_round=per_round(ler, r['r'])))
    with open(os.path.join(a.out, 'ler_table.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)

    ds = sorted({t['d'] for t in table})
    ps = sorted({t['p'] for t in table})
    get = lambda d, key: np.array([next(t[key] for t in table if t['d'] == d and t['p'] == p)
                                   for p in ps])
    cross = {}
    for d1, d2 in zip(ds, ds[1:]):
        cross[f'd{d1}_d{d2}'] = dict(
            per_shot=crossing(ps, get(d1, 'ler'), get(d2, 'ler')),
            per_round=crossing(ps, get(d1, 'ler_per_round'), get(d2, 'ler_per_round')))
    cross['fowler_reference'] = FOWLER
    with open(os.path.join(a.out, 'crossings.json'), 'w') as fh:
        json.dump(cross, fh, indent=1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for d in ds:
        y, lo, hi = get(d, 'ler'), get(d, 'ci95_lo'), get(d, 'ci95_hi')
        ax.errorbar(ps, y, yerr=[y - lo, hi - y], marker='o', ms=4, capsize=2, label=f'd={d}')
    ax.axvline(FOWLER, color='gray', ls=':', lw=1, label='Fowler 0.57%')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('physical error rate p'); ax.set_ylabel('logical error rate per shot (r = d)')
    ax.set_title('MWPM, rotated surface-code Z memory, uniform depolarizing noise')
    ax.legend(); ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'threshold.png'), dpi=150)

    print(f"{'d':>2} {'p':>7} {'shots':>11} {'errors':>7} {'LER':>10} {'95% CI':>23}")
    for t in table:
        print(f"{t['d']:>2} {t['p']:>7} {t['shots']:>11,} {t['errors']:>7,} {t['ler']:>10.3e} "
              f"[{t['ci95_lo']:.3e}, {t['ci95_hi']:.3e}]")
    print(json.dumps(cross, indent=1))


if __name__ == '__main__':
    main()
