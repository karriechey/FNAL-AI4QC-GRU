# Created: 2026-10-27
# Last modified: 2026-10-27
# Threshold figure: MWPM logical error rate vs p from the sinter sweep, GRU test-set results
# as stars. Every point is a counted simulation result; the dashed fit lines are a guide only.
#   python figures/threshold_figure.py
import csv
import json
from collections import defaultdict
from math import ceil, log
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "experiments" / "threshold_sweep_2026-10-27"
OUT = ROOT / "figures" / "output"

MIN_ERRORS = 100                  # points below this are left out of the plot and the fit
BAND_LIKELIHOOD_RATIO = 1000      # same band definition as sinter's plots
OPERATING_P = 0.004

COLORS = {3: "#4C72B0", 5: "#DD8452", 7: "#55A868", 9: "#C44E52", 11: "#8172B3"}
STYLES = {3: "-", 5: "-.", 7: ":", 9: (0, (3, 1, 1, 1, 1, 1)), 11: (0, (8, 2, 1, 2))}

# GRU test-set rows: (d, p) -> (result file, row filter). The d9 p=0.004 headline is f3280e240;
# f7w_p004 was scored by its launcher, not by eval_selected.sh (protocol/selection_rule.md).
GRU_SOURCES = {
    (5, 0.003): (ROOT / "protocol/eval_selected.csv", "f5w_p003_"),
    (5, 0.004): (ROOT / "protocol/eval_selected.csv", "f5w_p004_"),
    (5, 0.005): (ROOT / "protocol/eval_selected.csv", "f5w_p005_"),
    (7, 0.003): (ROOT / "protocol/eval_selected.csv", "f7w_p003_"),
    (7, 0.004): (SWEEP / "gru_test_f7w_p004_eval_best_ckpt.csv", "f7w_p004_"),
    (7, 0.005): (ROOT / "protocol/eval_selected.csv", "f7w_p005_"),
    (9, 0.003): (ROOT / "protocol/eval_selected.csv", "f9w_p003_"),
    (9, 0.004): (ROOT / "protocol/eval_selected.csv", "f3280e240_"),
    (9, 0.005): (ROOT / "protocol/eval_selected.csv", "f9w_p005_"),
}


def load_sweep(path):
    # sinter appends one row per batch; sum them per (d, p)
    agg = defaultdict(lambda: [0, 0, 0.0])
    for r in csv.DictReader(open(path), skipinitialspace=True):
        m = json.loads(r["json_metadata"])
        a = agg[m["d"], m["p"]]
        a[0] += int(r["shots"])
        a[1] += int(r["errors"])
        a[2] += float(r["seconds"])
    return {k: dict(shots=v[0], errors=v[1], cpu_s=v[2]) for k, v in agg.items()}


def likelihood_band(errors, shots, ratio=BAND_LIKELIHOOD_RATIO):
    p_hat = errors / shots
    ll = lambda q: errors * log(q) + (shots - errors) * log(1 - q)
    target = ll(p_hat) - log(ratio)
    lo = brentq(lambda q: ll(q) - target, 1e-15, p_hat) if errors > 0 else 0.0
    hi = brentq(lambda q: ll(q) - target, p_hat, 1 - 1e-12)
    return lo, hi


def fit_threshold(points):
    # log P_L - k log p = log A - k log p_th, k = ceil(d/2): linear least squares in (log A, log p_th),
    # weighted by the error count (var of log P_L ~ 1/errors); refit on points below the new p_th
    p_th = 0.01
    for _ in range(50):
        rows = [(d, p, v) for (d, p), v in points.items() if p < p_th]
        k = np.array([ceil(d / 2) for d, _, _ in rows], float)
        y = np.array([log(v["errors"] / v["shots"]) - kk * log(p) for (_, p, v), kk in zip(rows, k)])
        w = np.sqrt([v["errors"] for _, _, v in rows])
        X = np.column_stack([np.ones_like(k), -k])
        (log_a, log_pth), *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
        if abs(np.exp(log_pth) - p_th) < 1e-9:
            break
        p_th = float(np.exp(log_pth))
    return float(np.exp(log_a)), p_th, len(rows)


def load_gru():
    stars = []
    for (d, p), (path, run_prefix) in GRU_SOURCES.items():
        hits = [r for r in csv.DictReader(open(path))
                if f"/{run_prefix}" in r["weights_path"] and int(r["d"]) == d
                and abs(float(r["p"]) - p) < 1e-9]
        if len(hits) != 1:
            raise SystemExit(f"{run_prefix} d={d} p={p}: {len(hits)} rows in {path}")
        r = hits[0]
        stars.append(dict(d=d, p=p, gru=float(r["p_L"]), mwpm=float(r["mwpm_p_L"]),
                          n=int(r["n_test"]), source=str(path.relative_to(ROOT)),
                          weights=r["weights_path"]))
    return stars


def setup_style():
    for f in font_manager.findSystemFonts():
        if "cmun" in Path(f).name.lower():
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({
        "font.family": "CMU Serif", "mathtext.fontset": "cm", "axes.unicode_minus": False,
        "xtick.labelsize": 18, "ytick.labelsize": 18, "axes.labelsize": 22, "legend.fontsize": 18,
        "axes.linewidth": 1, "xtick.major.width": 1, "ytick.major.width": 1,
        "xtick.minor.width": 1, "ytick.minor.width": 1,
    })


def plot(points, stars, a_fit, p_th):
    fig, ax = plt.subplots(figsize=(14.25, 6.0))
    ax.axvline(OPERATING_P, color="0.6", lw=0.8, zorder=0)
    p_guide = np.geomspace(1e-3, p_th, 100)
    for d in sorted({d for d, _ in points}):
        ps = sorted(p for dd, p in points if dd == d)
        v = [points[d, p] for p in ps]
        pl = [x["errors"] / x["shots"] for x in v]
        band = np.array([likelihood_band(x["errors"], x["shots"]) for x in v])
        c = COLORS[d]
        ax.fill_between(ps, band[:, 0], band[:, 1], color=c, alpha=0.18, lw=0, zorder=1)
        ax.plot(ps, pl, color=c, ls=STYLES[d], lw=1.8, marker="o", ms=4, label=f"$d = {d}$", zorder=2)
        ax.plot(p_guide, a_fit * (p_guide / p_th) ** ceil(d / 2), color=c, ls=(0, (5, 4)), lw=1.0,
                alpha=0.8, zorder=1)
    for s in stars:
        ax.plot(s["p"], s["gru"], marker="*", ms=22, color=COLORS[s["d"]], mec="black", mew=0.8,
                ls="none", zorder=4)

    handles, labels = ax.get_legend_handles_labels()
    handles += [plt.Line2D([], [], marker="*", ms=22, color="0.35", mec="black", mew=0.8, ls="none"),
                plt.Line2D([], [], color="0.35", ls=(0, (5, 4)), lw=1.0)]
    labels += ["GRU", "fit"]
    ax.legend(handles, labels, loc="upper left", ncol=2, frameon=False, handlelength=2.6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(9e-4, 1.65e-2)
    x_ticks = [0.001, 0.002, 0.004, 0.008, 0.015]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{t:g}" for t in x_ticks])
    ax.set_xlabel("physical error rate $p$")
    ax.set_ylabel("logical error rate $P_L$")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    sweep = load_sweep(SWEEP / "sinter_stats.csv")
    points = {k: v for k, v in sweep.items() if v["errors"] >= MIN_ERRORS}
    skipped = sorted(k for k in sweep if k not in points)
    a_fit, p_th, n_fit = fit_threshold(points)
    stars = load_gru()

    setup_style()
    fig = plot(points, stars, a_fit, p_th)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_threshold.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "fig_threshold_transparent.png", dpi=300, transparent=True)

    summary = dict(p_th=p_th, A=a_fit, points_in_fit=n_fit, skipped_below_100_errors=skipped,
                   total_shots=sum(v["shots"] for v in sweep.values()),
                   total_cpu_s=sum(v["cpu_s"] for v in sweep.values()),
                   gru=[dict(s, ratio=s["gru"] / s["mwpm"]) for s in stars])
    json.dump(summary, open(OUT / "fig_threshold_summary.json", "w"), indent=1, default=list)
    print(f"p_th = {p_th:.5f}  A = {a_fit:.4f}  ({n_fit} points in fit); skipped {skipped}")
    for s in stars:
        print(f"d={s['d']} p={s['p']}: GRU {s['gru']:.6f} MWPM {s['mwpm']:.6f} "
              f"ratio {s['gru'] / s['mwpm']:.3f}  {s['source']}  {s['weights']}")
