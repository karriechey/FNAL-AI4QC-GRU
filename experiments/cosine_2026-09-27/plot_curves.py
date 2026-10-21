# Created: 2026-09-27
# Last modified: 2026-09-28 (training loss on a log axis so the cosine runs stay in view; clipped fraction as markers; shorter panel title)

"""Training curves of the cosine-schedule queue, with the constant-LR run M2 for reference.

Panels: training loss, validation loss, validation LER, LR and fraction of clipped updates.
Writes loss_curves.png/.pdf and curves_summary.json.

python3 plot_curves.py        (from experiments/cosine_2026-09-27)
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
RUNS = [  # label, glob of history.json, colour, line style
    ('u280 50M s0, constant 3e-3 (M2)',
     f'{EXP}/morning_2026-09-27/M2_d7_u280_50M_s0_*/seed0/runs/*.history.json', '#8a8a8a', '-'),
    ('u280 50M s0, cosine + clip (K1)', f'{HERE}/K1_d7_u280_50M_s0_*/seed0/runs/*.history.json', '#2a78d6', '-'),
    ('u280 50M s1, cosine + clip (K2)', f'{HERE}/K2_d7_u280_50M_s1_*/seed1/runs/*.history.json', '#2a78d6', '--'),
    ('u560 50M s0, cosine + clip (K3)', f'{HERE}/K3_d7_u560_50M_s0_*/seed0/runs/*.history.json', '#e34948', '-'),
]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    summary = {}
    for label, pattern, col, ls in RUNS:
        files = sorted(f for f in glob.glob(pattern) if '.superseded_' not in f)
        if not files:
            continue
        h = json.load(open(files[-1]))
        ep = np.arange(1, len(h['loss']) + 1)
        tl = np.array(h['loss'])
        ax[0, 0].plot(ep, tl, color=col, ls=ls, lw=1.3, label=label)
        ax[0, 1].plot(ep, h['val_loss'], color=col, ls=ls, lw=1.1, label=label)
        ax[1, 0].plot(ep, h['val_ler'], color=col, ls=ls, lw=1.1, label=label)
        if 'lr' in h:
            ax[1, 1].plot(ep, h['lr'], color=col, ls=ls, lw=1.1, label=f'{label}: learning rate')
        if 'clip_frac' in h:
            cf = np.array(h['clip_frac'])
            nz = cf > 0          # epochs with no clipped update are left out of the log axis
            ax[1, 1].plot(ep[nz], cf[nz], color=col, ls='none',
                          marker='o' if ls == '-' else 's', ms=3,
                          label=f'{label}: fraction of updates clipped')
        i_min = int(np.argmin(tl))
        summary[label] = dict(
            history=os.path.relpath(files[-1], EXP), epochs=int(len(tl)),
            best_val_ler_epoch=int(np.argmin(h['val_ler']) + 1), min_val_ler=float(np.min(h['val_ler'])),
            train_loss_min=float(tl[i_min]), train_loss_min_epoch=i_min + 1,
            train_loss_last=float(tl[-1]), rise_min_to_last=float(tl[-1] - tl[i_min]),
            clip_frac_cum_last=(float(h['clip_frac_cum'][-1]) if 'clip_frac_cum' in h else None))
    ax[0, 0].set_yscale('log'); ax[0, 0].set_ylim(0.002, 0.2)
    ax[0, 1].set_ylim(0.012, 0.05)
    ax[1, 0].set_ylim(0.005, 0.015)
    ax[1, 1].set_yscale('log')
    for a, t, y in ((ax[0, 0], 'training loss', 'binary cross-entropy'),
                    (ax[0, 1], 'validation loss', 'binary cross-entropy'),
                    (ax[1, 0], 'validation logical error rate', 'logical error rate'),
                    (ax[1, 1], 'learning rate, clipped updates', 'value')):
        a.set_title(f'd=7, 50M training shots: {t}'); a.set_xlabel('training epoch')
        a.set_ylabel(y); a.grid(alpha=.3); a.legend(fontsize=7)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(HERE, f'loss_curves.{ext}'), dpi=130)
    json.dump(summary, open(os.path.join(HERE, 'curves_summary.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
