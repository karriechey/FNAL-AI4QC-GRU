# Created: 2026-09-27
# Last modified: 2026-09-27
"""Choose global_clipnorm from the probe's per-update gradient norms (rule fixed in advance).

    python3 pick_threshold.py <grad_norms.npy> --out threshold.json

threshold = 5 x median(norms of updates 100..500, 1-indexed inclusive), rounded to 2
significant figures. Also records the distribution of all 500 norms and of updates
100..500, and the fraction of the probe's updates 100..500 that the threshold would have
clipped. Draws grad_norms_probe.png (norm per update, with the median and the threshold).
"""
import argparse
import json
import os

import numpy as np


def round_sig(x, sig=2):
    return float(f'{x:.{sig}g}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('norms')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    n = np.load(a.norms)
    assert len(n) >= 500, f'probe has only {len(n)} updates'
    win = n[99:500]                       # updates 100..500 (1-indexed), 401 values
    med = float(np.median(win))
    thr = round_sig(5 * med)
    q = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    res = dict(
        rule='global_clipnorm = 5 x median(updates 100-500), 2 significant figures',
        n_updates=int(len(n)), median_100_500=med, five_x_median=5 * med,
        global_clipnorm=thr,
        percentiles_all={str(k): float(v) for k, v in zip(q, np.percentile(n, q))},
        percentiles_100_500={str(k): float(v) for k, v in zip(q, np.percentile(win, q))},
        max_to_median_100_500=float(win.max() / med),
        frac_all_above_threshold=float(np.mean(n > thr)),
        frac_100_500_above_threshold=float(np.mean(win > thr)),
        first_10=[float(v) for v in n[:10]])
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(n) + 1), n, lw=1, color='#2a78d6', label='global gradient norm')
    ax.axhline(med, color='#555555', lw=1, ls='--', label=f'median of updates 100-500 = {med:.3g}')
    ax.axhline(thr, color='#e34948', lw=1.2, label=f'global_clipnorm = {thr:g}')
    ax.axvspan(100, 500, color='#999999', alpha=0.12)
    ax.set_yscale('log'); ax.set_xlabel('optimizer update'); ax.set_ylabel('global gradient norm')
    ax.set_title('d=7 u280 50M seed 0, cosine LR, no clipping: first 500 updates')
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(a.out), 'grad_norms_probe.png'), dpi=130)


if __name__ == '__main__':
    main()
