# Created: 2026-09-29
# Last modified: 2026-09-29
"""Overview figure of every GRU run so far (d5/d7, p = 0.004, r = d).

    python3 plot_overview.py          (writes overview.png and overview.pdf next to this file)

y axis everywhere: GRU logical error rate / MWPM logical error rate on the same 1.8M
evaluation shots [15.2M, 17M) of the FORMAL pool (1 = MWPM parity, lower is better).
Values are the best-validation-LER checkpoints listed in GRU_summary_updated_2026-09-29
(two older d7 u140 20M runs were selected by validation loss). Hollow markers: runs that
were still improving at their last epoch or became unstable at constant LR 3e-3, so their
number is a limit of that budget, not a converged value.

Panels
  (a) data scaling at d7: training shots 20M / 50M / 100M, per width and LR schedule
  (b) width scaling at d7: parameter count, at 20M and 50M
  (c) learning rate at d7 u140 20M, constant LR
  (d) GRU reset_after=False vs True
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
P = {'u70': 25061, 'u140': 79521, 'u200': 149601, 'u280': 276641, 'u560': 1023681}


def style(ax, title):
    ax.set_title(title, loc='left', fontsize=10.5, color=INK)
    ax.axhline(1.0, color=INK2, lw=1, ls='--', zorder=1)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=INK2, labelsize=8.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)
    ax.set_ylabel('GRU p_L / MWPM p_L', fontsize=9, color=INK2)


def pts(ax, x, ys, color, marker, hollow=False, label=None):
    """Each seed as its own marker; hollow = not converged or unstable."""
    for i, y in enumerate(np.atleast_1d(ys)):
        ax.scatter([x], [y], s=46, marker=marker, zorder=3, linewidths=1.6,
                   facecolors='white' if hollow else color, edgecolors=color,
                   label=label if i == 0 else None)


def mwpm_label(ax, x):
    ax.text(x, 1.0, ' MWPM', va='bottom', ha='left', fontsize=8, color=INK2)


fig, axs = plt.subplots(2, 2, figsize=(12, 9))

# (a) data scaling, d7
ax = axs[0, 0]
series = [  # label, colour, marker, {shots M: [values per seed]}
    ('u140, constant 3e-3', BLUE, 'o', {20: [2.261, 1.956], 50: [2.211]}),
    ('u280, constant 3e-3', ORANGE, 's', {20: [1.903, 1.781], 50: [1.453]}),
    ('u280, cosine + clip', AQUA, 'D', {20: [1.934], 50: [1.250, 1.243], 100: [1.062]}),
    ('u560, cosine + clip', YELLOW, '^', {50: [1.388]}),
]
for lab, c, m, d in series:
    xs = sorted(d)
    ax.plot(xs, [np.mean(d[x]) for x in xs], color=c, lw=2, zorder=2)
    for x in xs:
        pts(ax, x, d[x], c, m, label=lab if x == xs[0] else None)
    ax.annotate(lab, (xs[-1], np.mean(d[xs[-1]])), xytext=(7, 0), textcoords='offset points',
                va='center', fontsize=8.5, color=INK)
ax.set_xscale('log'); ax.set_xticks([20, 50, 100]); ax.set_xticklabels(['20M', '50M', '100M'])
ax.set_xlim(16, 200); ax.set_ylim(0.9, 2.45)
ax.set_xlabel('training shots', fontsize=9, color=INK2)
style(ax, '(a) More data helps most with the cosine schedule (d=7)')
mwpm_label(ax, 17)

# (b) width scaling, d7
ax = axs[0, 1]
w20 = [('u70', [3.448], True), ('u140', [2.261, 1.956], False), ('u200', [2.114], True),
       ('u280', [1.903, 1.781], False)]
ax.plot([P[u] for u, _, _ in w20], [np.mean(v) for _, v, _ in w20], color=BLUE, lw=2, zorder=2)
for u, v, h in w20:
    pts(ax, P[u], v, BLUE, 'o', hollow=h)
ax.annotate('20M, constant 3e-3', (P['u280'], np.mean(w20[-1][1])), xytext=(7, 0),
            textcoords='offset points', va='center', fontsize=8.5, color=INK)
w50 = [('u140', [2.211]), ('u280', [1.453])]
ax.plot([P[u] for u, _ in w50], [v[0] for _, v in w50], color=ORANGE, lw=2, zorder=2)
for u, v in w50:
    pts(ax, P[u], v, ORANGE, 's')
ax.annotate('50M, constant 3e-3', (P['u280'], 1.453), xytext=(7, 0), textcoords='offset points',
            va='center', fontsize=8.5, color=INK)
w50c = [('u280', [1.250, 1.243]), ('u560', [1.388])]
ax.plot([P[u] for u, _ in w50c], [np.mean(v) for _, v in w50c], color=AQUA, lw=2, zorder=2)
for u, v in w50c:
    pts(ax, P[u], v, AQUA, 'D')
ax.annotate('50M, cosine + clip', (P['u560'], 1.388), xytext=(7, 0), textcoords='offset points',
            va='center', fontsize=8.5, color=INK)
for u in P:
    ax.annotate(u, (P[u], 0.93), ha='center', fontsize=7.5, color=INK2)
ax.set_xscale('log'); ax.set_xlim(1.8e4, 4e6); ax.set_ylim(0.9, 3.7)
ax.set_xlabel('parameters', fontsize=9, color=INK2)
style(ax, '(b) Wider helps up to u280; u560 overfits at 50M (d=7)')
mwpm_label(ax, 1.9e4)

# (c) learning rate, d7 u140 20M constant, 200 epochs
ax = axs[1, 0]
lr = [(3e-5, [7.923], True), (3e-4, [3.082], True), (3e-3, [2.261, 1.956], False)]
ax.plot([x for x, _, _ in lr], [np.mean(v) for _, v, _ in lr], color=BLUE, lw=2, zorder=2)
for x, v, h in lr:
    pts(ax, x, v, BLUE, 'o', hollow=h)
ax.annotate('still improving at epoch 200', (3e-5, 7.923), xytext=(8, -2), textcoords='offset points',
            fontsize=8, color=INK2, va='center')
ax.annotate('still improving at epoch 200', (3e-4, 3.082), xytext=(8, 6), textcoords='offset points',
            fontsize=8, color=INK2)
ax.set_xscale('log'); ax.set_xticks([3e-5, 3e-4, 3e-3]); ax.set_xticklabels(['3e-5', '3e-4', '3e-3'])
ax.set_xlim(1.5e-5, 1.2e-2); ax.set_ylim(0.9, 8.6)
ax.set_xlabel('constant learning rate (u140, 20M, 200 epochs)', fontsize=9, color=INK2)
style(ax, '(c) Lower constant LR is far slower within 200 epochs (d=7)')
mwpm_label(ax, 1.6e-5)

# (d) reset_after
ax = axs[1, 1]
groups = [('d=5\n20M', [0.939], [1.049, 1.056, 1.052]),
          ('d=7\n20M', [2.261, 1.956], [2.416, 2.417]),
          ('d=7\n50M', [2.211], [2.497])]
for i, (g, f, t) in enumerate(groups):
    pts(ax, i - 0.12, f, BLUE, 'o', label='reset_after=False' if i == 0 else None)
    pts(ax, i + 0.12, t, ORANGE, 's', label='reset_after=True' if i == 0 else None)
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g for g, _, _ in groups])
ax.set_xlim(-0.6, 2.6); ax.set_ylim(0.8, 2.7)
ax.legend(frameon=False, fontsize=8.5, loc='upper left')
style(ax, '(d) reset_after=False is better at every point (u140, constant 3e-3)')
mwpm_label(ax, -0.58)

fig.text(0.01, 0.005,
         'p = 0.004, r = d, batch 10,000, seed 0 (+ seed 1/2 where two or three markers share an x). '
         'Evaluation on the same 1.8M non-sealed shots; MWPM decoded on those shots. Best checkpoint by validation LER. '
         'Lines join seed means. Hollow = still improving at the last epoch or unstable at constant 3e-3 (budget-limited).',
         fontsize=7.8, color=INK2, wrap=True)
fig.tight_layout(rect=(0, 0.03, 1, 1))
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(HERE, f'overview.{ext}'), dpi=150, facecolor='white')
print('wrote overview.png/.pdf')
