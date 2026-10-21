# Created: 2026-10-21
# Last modified: 2026-10-21

"""Depth scaling at d=7 from the newest GRU_depth_scaling_runs_*.csv.

python figures/depth_scaling.py   -> figures/output/depth_scaling.{pdf,png}, lr_stability_vs_depth.{pdf,png}
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scaling_figs import N_EVAL, OUT_DIR, REPO, wilson  # also applies the shared rcParams

STACK = "#D55E00"
SINGLE = "#0072B2"
STATUS = {"stable": ("o", "#009E73"), "unstable": ("s", "#E69F00"), "diverged": ("X", "#CC3311")}
L1_PARAMS, PER_LAYER = 25061, 29610


def load():
    path = sorted(REPO.glob("GRU_depth_scaling_runs_*.csv"))[-1]
    return path, list(csv.DictReader(open(path)))


def ratio_ci(r):
    lo, hi = wilson(float(r["p_L"]) * N_EVAL, N_EVAL)
    return float(r["Relative to MWPM"]), lo / float(r["MWPM"]), hi / float(r["MWPM"])


def depth_fig(rows):
    done = [r for r in rows if r["Part"] == "B" and r["Status"] == "Done"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    stacks = sorted((r for r in done if r["Width"] == "u70"), key=lambda r: int(r["L"]))
    if stacks:
        pts = [(int(r["L"]), *ratio_ci(r)) for r in stacks]
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                    yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]],
                    fmt="-o", ms=6, lw=2, color=STACK, label="L x GRU u70")
    for r in done:
        if r["Width"] != "u70":
            y, lo, hi = ratio_ci(r)
            ax.errorbar(int(r["L"]) + 0.25, y, yerr=[[y - lo], [hi - y]], fmt="s", ms=8, mfc="white",
                        mew=2, color=SINGLE, label="1 x GRU, matched parameters")
            ax.annotate(r["Width"], (int(r["L"]) + 0.25, y), xytext=(7, 0), textcoords="offset points",
                        fontsize=8, va="center")
    ax.axhline(1.0, color="black", lw=1, ls="--")
    ax.text(10.4, 1.0, "MWPM", fontsize=8, va="bottom", ha="right")
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("GRU layers L")
    ax.set_ylabel("logical error rate / MWPM (eval block)")
    ax.set_title("Depth at fixed width, d = 7, p = 0.004")
    top = ax.secondary_xaxis("top", functions=(lambda L: (L1_PARAMS + PER_LAYER * (L - 1)) / 1e3,
                                               lambda k: (k * 1e3 - L1_PARAMS) / PER_LAYER + 1))
    top.set_xticks([25, 100, 200, 290])
    top.set_xlabel("parameters (thousands, u70 stack)")
    handles, labels = ax.get_legend_handles_labels()
    keep = dict(zip(labels, handles))
    ax.legend(keep.values(), keep.keys(), loc="upper right")
    fig.tight_layout()
    return fig


def lr_fig(rows):
    scan = [r for r in rows if r["Part"] == "A"]
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    for status, (marker, color) in STATUS.items():
        pts = [r for r in scan if r["Status"] == status]
        if pts:
            ax.plot([int(r["L"]) for r in pts], [float(r["eta x N_det"]) for r in pts], ls="", marker=marker,
                    ms=9, color=color, mec="white", mew=1, label=status)
    ax.set_yscale("log")
    ax.set_xticks([1, 10])
    ax.set_xlim(0, 11)
    ax.set_yticks([0.17, 0.34, 0.67, 1.0, 1.7])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("GRU layers L (u70)")
    ax.set_ylabel("peak LR x N_det (N_det = 336)")
    ax.set_title("LR stability over 20 epochs, d = 7")
    ax.legend(loc="center")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    path, rows = load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in (("depth_scaling", depth_fig(rows)), ("lr_stability_vs_depth", lr_fig(rows))):
        fig.savefig(OUT_DIR / f"{name}.pdf")
        fig.savefig(OUT_DIR / f"{name}.png", dpi=300)
    print(f"{path.name}: {len(rows)} rows -> {OUT_DIR}/depth_scaling.png, lr_stability_vs_depth.png")
