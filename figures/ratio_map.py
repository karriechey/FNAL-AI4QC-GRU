# Created: 2026-10-28
# Last modified: 2026-10-28 (sentence-case axis labels)
# GRU / MWPM logical error rate ratio over d x p (test-set cells, the threshold figure's stars),
# and the suppression factor Lambda at p = 0.004 for both decoders.
#   python figures/ratio_map.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import poster_style
from threshold_figure import POSTER, load_gru

DISTANCES = (5, 7, 9)
PS = (0.003, 0.004, 0.005)
LAMBDA_P = 0.004
FRAME_IN = (6.9, 3.0)
# GRU better -> white at parity -> MWPM better, from the poster's GRU and MWPM colours
RATIO_CMAP = LinearSegmentedColormap.from_list("ratio", [poster_style.MODEL, "white", poster_style.NEUTRAL])


def per_round(p_l, rounds):
    return (1 - (1 - 2 * p_l) ** (1 / rounds)) / 2


def fit_lambda(d, p_l, n):
    # log eps_d = log C - (d+1)/2 log Lambda, weighted by binomial errors propagated to log eps_d
    eps = per_round(p_l, d)
    deps_dp = (1 - 2 * p_l) ** (1 / d - 1) / d
    sigma_log = deps_dp * np.sqrt(p_l * (1 - p_l) / n) / eps
    X = np.column_stack([np.ones_like(d, float), -(d + 1) / 2])
    w = 1 / sigma_log
    coef, *_ = np.linalg.lstsq(X * w[:, None], np.log(eps) * w, rcond=None)
    cov = np.linalg.inv((X * w[:, None]).T @ (X * w[:, None]))
    chi2 = float((((X @ coef) - np.log(eps)) * w) ** 2 @ np.ones_like(w))
    lam = np.exp(coef[1])
    return dict(Lambda=lam, err=lam * np.sqrt(cov[1, 1]), chi2=chi2, dof=len(d) - 2)


def plot(cells, lam):
    fig = plt.figure(figsize=FRAME_IN)
    ax = fig.add_axes([0.72 / 6.9, 0.62 / 3.0, 3.3 / 6.9, 2.28 / 3.0])
    ratio = np.full((len(DISTANCES), len(PS)), np.nan)
    for (d, p), s in cells.items():
        ratio[DISTANCES.index(d), PS.index(p)] = s["gru"] / s["mwpm"]
    ax.imshow(np.ma.masked_invalid(ratio), cmap=RATIO_CMAP, norm=TwoSlopeNorm(1, 0.5, 1.5),
              origin="lower", aspect="auto")
    for i, d in enumerate(DISTANCES):
        for j, p in enumerate(PS):
            if np.isnan(ratio[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="//",
                                           ec=poster_style.NEUTRAL, lw=0))
                ax.text(j, i, "pending", ha="center", va="center", fontsize=poster_style.TICK_PT,
                        color=poster_style.INK, bbox=dict(fc="white", ec="none", pad=1))
            else:
                ax.text(j, i, f"{ratio[i, j]:.2f}", ha="center", va="center",
                        fontsize=poster_style.CURVE_LABEL_PT, color=poster_style.INK)
    ax.set_xticks(range(len(PS)))
    ax.set_xticklabels([f"{p:g}" for p in PS])
    ax.set_yticks(range(len(DISTANCES)))
    ax.set_yticklabels([str(d) for d in DISTANCES])
    ax.tick_params(length=0)
    ax.set_xlabel("Physical error rate $p$")
    ax.set_ylabel("Code distance $d$")

    x0 = 4.45 / 6.9
    fig.text(x0, 0.80, f"$\\Lambda$ at $p$ = {LAMBDA_P:g}", fontsize=poster_style.AXIS_LABEL_PT,
             color=poster_style.INK, va="center")
    for y, name, color in ((0.56, "MWPM", poster_style.NEUTRAL), (0.34, "GRU", poster_style.MODEL)):
        fig.text(x0, y, f"{name}  {lam[name]['Lambda']:.2f} $\\pm$ {lam[name]['err']:.2f}",
                 fontsize=poster_style.AXIS_LABEL_PT, color=color, va="center")
    return fig


if __name__ == "__main__":
    cells = {(s["d"], s["p"]): s for s in load_gru() if s["d"] in DISTANCES and s["p"] in PS}
    rows = [cells[d, LAMBDA_P] for d in DISTANCES]
    d = np.array(DISTANCES)
    n = np.array([r["n"] for r in rows])
    lam = {name: fit_lambda(d, np.array([r[key] for r in rows]), n)
           for name, key in (("MWPM", "mwpm"), ("GRU", "gru"))}

    poster_style.apply()
    fig = plot(cells, lam)
    POSTER.mkdir(parents=True, exist_ok=True)
    fig.savefig(POSTER / "fig6_ratio_map.png", dpi=300, facecolor="white")
    fig.savefig(POSTER / "fig6_ratio_map_transparent.png", dpi=300, transparent=True)
    for (dd, p), s in sorted(cells.items()):
        print(f"d={dd} p={p}: ratio {s['gru'] / s['mwpm']:.4f}  {s['source']}")
    for (dd, p) in [(dd, p) for dd in DISTANCES for p in PS if (dd, p) not in cells]:
        print(f"d={dd} p={p}: pending")
    for name, v in lam.items():
        print(f"Lambda {name} = {v['Lambda']:.3f} +- {v['err']:.3f}  chi2/dof = {v['chi2']:.2f}/{v['dof']}")
