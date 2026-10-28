# Created: 2026-10-28
# Last modified: 2026-10-28
# Poster version of depth_scaling.py: d = 7, p = 0.004, u70 stacks L = 1-10 and the single-layer
# u172 / u255 runs matched to L = 4 / 8 in parameters. Eval-block ratios to MWPM on the same shots.
#   python figures/fig7_depth.py
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import poster_style

ROOT = Path(__file__).resolve().parents[1]
POSTER = ROOT / "figures" / "poster"
N_EVAL = 1_800_000
L1_PARAMS, PER_LAYER = 25061, 29610   # u70 stack at d = 7
MATCH_OFFSET = 0.3                    # matched squares drawn just right of their L
FRAME_IN = (4.5, 3.0)
MARKER_MS = 6


def load():
    path = sorted(ROOT.glob("GRU_depth_scaling_runs_*.csv"))[-1]
    rows = [r for r in csv.DictReader(open(path)) if r["Part"] == "B" and r["Status"] == "Done"]
    for r in rows:
        p_l, mwpm = float(r["p_L"]), float(r["MWPM"])
        r.update(L=int(r["L"]), ratio=p_l / mwpm, err=np.sqrt(p_l * (1 - p_l) / N_EVAL) / mwpm)
    return path, rows


def plot(rows):
    fig, ax = plt.subplots(figsize=FRAME_IN)
    stack = sorted((r for r in rows if r["Width"] == "u70"), key=lambda r: r["L"])
    ax.errorbar([r["L"] for r in stack], [r["ratio"] for r in stack], yerr=[r["err"] for r in stack],
                fmt="-o", ms=MARKER_MS, lw=2, color=poster_style.MODEL, elinewidth=1, zorder=3)
    for r in rows:
        if r["Width"] == "u70":
            continue
        x = r["L"] + MATCH_OFFSET
        ax.errorbar(x, r["ratio"], yerr=r["err"], fmt="s", ms=MARKER_MS + 1, mfc="white", mew=1.5,
                    color=poster_style.NEUTRAL, elinewidth=1, zorder=3)
        ax.annotate(r["Width"], (x, r["ratio"]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=poster_style.NEUTRAL, fontsize=poster_style.CURVE_LABEL_PT)

    l8 = next(r for r in stack if r["L"] == 8)
    u255 = next(r for r in rows if r["Width"] == "u255")
    # the L = 8 / u255 gap is too small to hold text, so the label sits above with an arrow into it
    ax.annotate("deep beats wide\nat equal parameters", (8 + MATCH_OFFSET / 2, np.sqrt(l8["ratio"] * u255["ratio"])),
                xytext=(7.4, 2.2), ha="center", va="center", fontsize=poster_style.CURVE_LABEL_PT,
                color=poster_style.INK, linespacing=1.0,
                arrowprops=dict(arrowstyle="-", color=poster_style.INK, lw=0.8, shrinkA=2, shrinkB=1))

    ax.axhline(1.0, color=poster_style.INK, lw=0.8, ls="--", zorder=1)
    ax.text(1.0, 1.0, "MWPM", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=poster_style.CURVE_LABEL_PT, color=poster_style.INK)

    ax.set_yscale("log")
    ax.set_ylim(0.6, 4.5)
    y_ticks = [0.7, 1, 2, 3]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{t:g}" for t in y_ticks])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(0.5, 10.9)
    ax.set_xticks([1, 2, 4, 6, 8, 10])
    ax.set_xlabel("GRU layers $L$")
    ax.set_ylabel("$P_L$ / MWPM")
    top = ax.secondary_xaxis("top", functions=(lambda L: (L1_PARAMS + PER_LAYER * (L - 1)) / 1e3,
                                               lambda k: (k * 1e3 - L1_PARAMS) / PER_LAYER + 1))
    top.set_xticks([25, 100, 200, 290])
    top.set_xlabel("parameters (thousands)")
    fig.tight_layout(pad=0.3)
    return fig


if __name__ == "__main__":
    path, rows = load()
    poster_style.apply()
    fig = plot(rows)
    POSTER.mkdir(parents=True, exist_ok=True)
    fig.savefig(POSTER / "fig7_depth.png", dpi=300, facecolor="white")
    fig.savefig(POSTER / "fig7_depth_transparent.png", dpi=300, transparent=True)

    # marker half-height in ratio units at this y position, for comparison with the error bars
    ax = fig.axes[0]
    to_px = ax.transData.transform
    print(path.name)
    for r in sorted(rows, key=lambda r: (r["Width"] != "u70", r["L"])):
        y_px = to_px((r["L"], r["ratio"]))[1]
        half_marker = ax.transData.inverted().transform((0, y_px + MARKER_MS / 2 * fig.dpi / 72))[1] - r["ratio"]
        print(f"{r['Run']:20s} L={r['L']:2d} {r['Width']:5s} params {r['Parameters']:>7s}  "
              f"ratio {r['ratio']:.4f} +- {r['err']:.4f}  (marker half-height {half_marker:.4f})")
