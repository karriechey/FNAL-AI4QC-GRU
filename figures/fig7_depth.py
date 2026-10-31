# Created: 2026-10-28
# Last modified: 2026-10-31 (depth annotation with arrow, heavier MWPM baseline line and label, SVG output; 2026-10-28: small-panel font sizes, MWPM label left, sentence-case labels, matched-run legend instead of bracket, 10 pt point labels)
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
LABEL_PT, TICK_PT, NOTE_PT = 13, 11, 11   # smaller than poster_style: this panel is 4.5 in wide
POINT_PT = 10
BASELINE_PT = 9


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
        # u255 sits just under the MWPM line, so its label goes to the left, raised to clear the L = 8 marker
        above = r["Width"] == "u172"
        ax.annotate(r["Width"], (x, r["ratio"]), xytext=(0, MARKER_MS) if above else (-MARKER_MS, 1),
                    textcoords="offset points", ha="center" if above else "right",
                    va="bottom", color=poster_style.NEUTRAL, fontsize=POINT_PT)

    key = plt.Line2D([], [], ls="none", marker="s", ms=MARKER_MS + 1, mfc="white", mew=1.5,
                     color=poster_style.NEUTRAL)
    ax.legend([key], ["1 wide GRU layer,\nsame parameters as the stack"], loc="upper right",
              frameon=False, fontsize=10, handletextpad=0.3, handlelength=1.0, borderaxespad=0.4)

    ax.axhline(1.0, color=poster_style.INK, lw=1.2, ls="--", zorder=1)
    ax.annotate("MWPM baseline", (0, 1.0), xycoords=ax.get_yaxis_transform(), xytext=(3, 1),
                textcoords="offset points", ha="left", va="bottom", fontsize=BASELINE_PT,
                color=poster_style.INK)

    # every u70 stack with L >= 4 is below the MWPM line; the arrow points at the first of them
    first_below = min((r for r in stack if r["ratio"] < 1), key=lambda r: r["L"])
    assert all(r["ratio"] < 1 for r in stack if r["L"] >= first_below["L"])
    note = ax.annotate(f"Increasing GRU depth improves\ndecoding performance, with\n"
                       f"\u2265{first_below['L']} layers outperforming MWPM.",
                       xy=(first_below["L"] + 0.15, first_below["ratio"] * 1.03), xytext=(5.0, 1.6),
                       ha="left", va="center", fontsize=POINT_PT, color=poster_style.INK,
                       arrowprops=dict(arrowstyle="-|>", lw=0.7, color=poster_style.NEUTRAL, shrinkA=3,
                                       shrinkB=5, mutation_scale=8))
    note.set_in_layout(False)   # keep the axes the same size as before the note

    ax.set_yscale("log")
    ax.set_ylim(0.6, 4.5)
    y_ticks = [0.7, 1, 2, 3]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{t:g}" for t in y_ticks])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(0.5, 10.9)
    ax.set_xticks([1, 2, 4, 6, 8, 10])
    ax.set_xlabel("GRU layers $L$")
    ax.set_ylabel("Error ratio (GRU / MWPM)")
    top = ax.secondary_xaxis("top", functions=(lambda L: (L1_PARAMS + PER_LAYER * (L - 1)) / 1e3,
                                               lambda k: (k * 1e3 - L1_PARAMS) / PER_LAYER + 1))
    top.set_xticks([25, 100, 200, 290])
    top.set_xlabel("Parameters (k)")
    fig.tight_layout(pad=0.3)
    return fig


if __name__ == "__main__":
    path, rows = load()
    poster_style.apply()
    plt.rcParams.update({"axes.labelsize": LABEL_PT, "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT})
    fig = plot(rows)
    POSTER.mkdir(parents=True, exist_ok=True)
    fig.savefig(POSTER / "fig7_depth.png", dpi=300, facecolor="white")
    fig.savefig(POSTER / "fig7_depth_transparent.png", dpi=300, transparent=True)
    fig.savefig(POSTER / "fig7_depth.svg", facecolor="white")

    # marker half-height in ratio units at this y position, for comparison with the error bars
    ax = fig.axes[0]
    to_px = ax.transData.transform
    print(path.name)
    for r in sorted(rows, key=lambda r: (r["Width"] != "u70", r["L"])):
        y_px = to_px((r["L"], r["ratio"]))[1]
        half_marker = ax.transData.inverted().transform((0, y_px + MARKER_MS / 2 * fig.dpi / 72))[1] - r["ratio"]
        print(f"{r['Run']:20s} L={r['L']:2d} {r['Width']:5s} params {r['Parameters']:>7s}  "
              f"ratio {r['ratio']:.4f} +- {r['err']:.4f}  (marker half-height {half_marker:.4f})")
