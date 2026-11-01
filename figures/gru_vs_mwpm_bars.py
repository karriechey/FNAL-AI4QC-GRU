# Created: 2026-10-27
# Last modified: 2026-10-28 (sentence-case axis label)
# GRU vs MWPM logical error rate at p = 0.004, d = 5, 7, 9: the selected checkpoints' test-set
# results (1.8M shots, MWPM on the same shots), the same rows as the threshold figure's stars.
#   python figures/gru_vs_mwpm_bars.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import poster_style
from threshold_figure import OUT, load_gru

P = 0.004
DISTANCES = (5, 7, 9)
FRAME_IN = (6.9, 3.0)


def std_err(p_l, n):
    return np.sqrt(p_l * (1 - p_l) / n)


def plot(rows):
    fig, ax = plt.subplots(figsize=FRAME_IN)
    x = np.arange(len(rows))
    width = 0.36
    gru = np.array([r["gru"] for r in rows])
    mwpm = np.array([r["mwpm"] for r in rows])
    n = np.array([r["n"] for r in rows])
    err_kw = dict(ecolor=poster_style.INK, elinewidth=1, capsize=3, capthick=1)
    ax.bar(x - width / 2, gru, width, yerr=std_err(gru, n), color=poster_style.MODEL, label="GRU",
           error_kw=err_kw, zorder=2)
    ax.bar(x + width / 2, mwpm, width, yerr=std_err(mwpm, n), color=poster_style.NEUTRAL, label="MWPM",
           error_kw=err_kw, zorder=2)
    for xi, g, m in zip(x, gru, mwpm):
        ax.annotate(f"{g / m:.2f}x", (xi - width / 2, g), textcoords="offset points", xytext=(0, 4),
                    ha="center", va="bottom", fontsize=poster_style.CURVE_LABEL_PT, color=poster_style.INK)

    ax.set_yscale("log")
    ax.set_ylim(5e-4, 2e-2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$d$ = {r['d']}" for r in rows])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Logical error rate $P_L$")
    ax.legend(loc="upper right", frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.3)
    return fig


if __name__ == "__main__":
    by_cell = {(s["d"], s["p"]): s for s in load_gru()}
    rows = [by_cell[d, P] for d in DISTANCES]
    poster_style.apply()
    fig = plot(rows)
    fig.savefig(OUT / "fig5_gru_vs_mwpm.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "fig5_gru_vs_mwpm_transparent.png", dpi=300, transparent=True)
    for r in rows:
        print(f"d={r['d']} p={r['p']}: GRU {r['gru']:.6f} +- {std_err(r['gru'], r['n']):.6f}  "
              f"MWPM {r['mwpm']:.6f} +- {std_err(r['mwpm'], r['n']):.6f}  ratio {r['gru'] / r['mwpm']:.3f}  "
              f"{r['source']}  {r['weights'].split('/experiments/')[-1]}")
