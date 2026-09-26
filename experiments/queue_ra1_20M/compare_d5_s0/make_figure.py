# Created: 2026-09-26
# Last modified: 2026-09-26 (three panels; evaluation-block panel removed)
"""Figure: d=5 GRU decoder, reset_after=False vs reset_after=True.

Every number is read from the run records on grace1 (histories, best-checkpoint JSONs,
evaluation CSVs, the MWPM validation-block JSON); nothing is typed in by hand.

    ~/tfenv/bin/python make_figure.py      # writes d5_reset_after_comparison.{png,pdf} here
"""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.dirname(HERE)
OLD = os.path.expanduser('~/rcnn_threshold/results/d5_u140_ntr20M_aug_20260821T154820Z')
BLUE, RED, GREEN = '#2a78d6', '#e34948', '#008300'   # validated palette slots (blue, red, green)
INK, MUTED = '#2b2b2a', '#6b6b67'


def csv_row(path):
    return list(csv.DictReader(open(path)))[-1]


# --- reset_after=False: old runs (best checkpoint = lowest validation loss) -------------
old = {}
for s in (0, 1, 2):
    h = json.load(open(glob.glob(f'{OLD}/seed{s}/runs/*.history.json')[0]))
    vl = h['val_loss']
    best = vl.index(min(vl)) + 1                  # Keras save_best_only keeps the first minimum
    ev = csv_row(f'{Q}/old_rescored_tf32off/d5_s{s}/eval_old_ra0.csv')
    old[s] = dict(loss=h['loss'], val_loss=vl, best=best, eval=float(ev['p_L']),
                  mwpm=float(ev['mwpm_p_L']))

# --- reset_after=True: new runs that have finished (best checkpoint = lowest val LER) ----
new = {}
for s in (0, 1, 2):
    d = sorted(glob.glob(f'{Q}/d5_s{s}_*/seed{s}'))
    if not d or not os.path.exists(f'{d[-1]}/COMPLETE'):
        continue
    d = d[-1]
    h = json.load(open(glob.glob(f'{d}/runs/*.history.json')[0]))
    b = json.load(open(glob.glob(f'{d}/ckpt/*.best.json')[0]))
    ev = csv_row(f'{d}/eval_best_ckpt.csv')
    new[s] = dict(loss=h['loss'], val_loss=h['val_loss'], val_ler=h['val_ler'],
                  best=int(b['best_epoch']), eval=float(ev['p_L']), mwpm=float(ev['mwpm_p_L']),
                  tf32=ev.get('tf32'))

mwpm_val = json.load(open(f'{HERE}/mwpm_val_block_d5.json'))['mwpm_p_L']
mwpm_eval = {r['mwpm'] for r in list(old.values()) + list(new.values())}
assert len(mwpm_eval) == 1, mwpm_eval       # every run was scored on the same eval shots
mwpm_eval = mwpm_eval.pop()

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False})
fig, (a, b, c) = plt.subplots(1, 3, figsize=(19, 6.2))
ep = list(range(1, 201))
lab_old = 'reset_after=False, 3 seeds (seed 0, seed 1, seed 2)'
lab_new = f"reset_after=True, {len(new)} seed{'s' if len(new) != 1 else ''} (" + \
    ', '.join(f'seed {s}' for s in sorted(new)) + ')'

# (a) training loss
for i, (s, r) in enumerate(sorted(old.items())):
    a.plot(ep, r['loss'], color=BLUE, lw=1.2, label=lab_old if i == 0 else None)
for i, (s, r) in enumerate(sorted(new.items())):
    a.plot(ep, r['loss'], color=RED, lw=1.2, label=lab_new if i == 0 else None)
a.set_ylim(0.015, 0.024)
a.set_title('(a)  Training loss', loc='left')
a.set_ylabel('binary cross-entropy loss')

# (b) validation loss with the selected checkpoints
for i, (s, r) in enumerate(sorted(old.items())):
    b.plot(ep, r['val_loss'], color=BLUE, lw=1.0, alpha=.85, label=lab_old if i == 0 else None)
for i, (s, r) in enumerate(sorted(new.items())):
    b.plot(ep, r['val_loss'], color=RED, lw=1.0, alpha=.85, label=lab_new if i == 0 else None)
for s, r in old.items():
    b.plot(r['best'], r['val_loss'][r['best'] - 1], 'o', ms=8, color=BLUE, mec='white', mew=1.5, zorder=5)
for s, r in new.items():
    b.plot(r['best'], r['val_loss'][r['best'] - 1], 'o', ms=8, color=RED, mec='white', mew=1.5, zorder=5)
b.set_ylim(0.0195, 0.0255)
b.set_title('(b)  Validation loss', loc='left')
b.set_ylabel('binary cross-entropy loss')
b.text(0.98, 0.97, 'dots: selected checkpoint\n'
       'reset_after=False: lowest validation loss\n'
       'reset_after=True: lowest validation LER',
       transform=b.transAxes, ha='right', va='top', fontsize=11, color=INK,
       bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#d9d8d4'))

# (c) validation LER, reset_after=True only
for i, (s, r) in enumerate(sorted(new.items())):
    c.plot(ep, r['val_ler'], color=RED, lw=1.0, alpha=.85, label=lab_new if i == 0 else None)
    c.plot(r['best'], r['val_ler'][r['best'] - 1], 'o', ms=8, color=RED, mec='white', mew=1.5, zorder=5)
    c.annotate(f"seed {s}: epoch {r['best']}", (r['best'], r['val_ler'][r['best'] - 1]),
               xytext=(0, 12), textcoords='offset points', ha='center', fontsize=11, color=INK,
               bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
c.axhline(mwpm_val, color=GREEN, ls='--', lw=1.6)
c.text(199, mwpm_val - 0.00003, 'MWPM (same shots)', color=INK, ha='right', va='top', fontsize=11)
c.set_ylim(0.0070, 0.0100)
c.set_title('(c)  Validation logical error rate', loc='left')
c.set_ylabel('logical error rate per shot')
c.text(0.98, 0.97, 'not logged by the reset_after=False runs\n'
       'dots: selected checkpoint (lowest validation LER)',
       transform=c.transAxes, ha='right', va='top', fontsize=11, color=MUTED)

for ax in (a, b, c):
    ax.set_xlabel('training epoch')
    ax.set_xlim(0, 202)
    ax.grid(alpha=.25)
handles, labels = a.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=2,
           frameon=False, fontsize=12, handlelength=2.5)

fig.suptitle('d=5 GRU decoder: reset_after=False vs reset_after=True', fontsize=15,
             fontweight='bold', color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.89))
for ext, kw in (('png', dict(dpi=200)), ('pdf', {})):
    fig.savefig(f'{HERE}/d5_reset_after_comparison.{ext}', **kw)

print('mwpm val block', mwpm_val, 'mwpm eval block', mwpm_eval)
for s, r in sorted(old.items()):
    print(f"reset_after=False seed {s}: best epoch {r['best']} (min val loss {min(r['val_loss']):.5f}), eval LER {r['eval']:.6f}")
for s, r in sorted(new.items()):
    print(f"reset_after=True  seed {s}: best epoch {r['best']} (val LER {min(r['val_ler']):.6f}), eval LER {r['eval']:.6f}, tf32 {r['tf32']}")
print('means x MWPM:', round(sum(r['eval'] for r in old.values()) / 3 / mwpm_eval, 4),
      round(sum(r['eval'] for r in new.values()) / len(new) / mwpm_eval, 4))
