# Created: 2026-10-27
# Last modified: 2026-10-31 (inset: dashed line at p = p_th with label; inset titled, p_th shown as 0.7%, inset axis labels spelled out; note reworded; GRU vs MWPM improvement note with arrow, lower-is-better label, SVG output; 2026-10-28: Greek nu from DejaVu Sans, MWPM legend entry; sentence-case axis labels; finite-size scaling collapse inset, PNGs to figures/poster; curve labels instead of distance legend, DIST colours; poster style: Helvetica, 18/15 pt; distance palette; d9 p=0.003 star removed; fit lines clipped to each d's data range)
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
import numpy as np
from scipy.optimize import brentq, least_squares

import poster_style

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "experiments" / "threshold_sweep_2026-10-27"
OUT = ROOT / "figures" / "output"
POSTER = ROOT / "figures" / "poster"

MIN_ERRORS = 100                  # points below this are left out of the plot and the fit
BAND_LIKELIHOOD_RATIO = 1000      # same band definition as sinter's plots
OPERATING_P = 0.004
COLLAPSE_WINDOW = (0.0056, 0.0092)  # p = 0.005603, 0.007167, 0.009168: the three sweep points around p_th

COLORS = poster_style.DIST
STAR_EDGE = "#1A1A1A"
STYLES = {3: "-", 5: "-.", 7: ":", 9: (0, (3, 1, 1, 1, 1, 1)), 11: (0, (8, 2, 1, 2))}

# GRU test-set rows: (d, p) -> (result file, row filter). The d9 p=0.004 headline is f3280e240;
# f7w_p004 was scored by its launcher, not by eval_selected.sh (protocol/selection_rule.md).
# d9 p=0.003 (f9w_p003) is left out: its headline model is undecided (240 epochs not converged).
GRU_SOURCES = {
    (5, 0.003): (ROOT / "protocol/eval_selected.csv", "f5w_p003_"),
    (5, 0.004): (ROOT / "protocol/eval_selected.csv", "f5w_p004_"),
    (5, 0.005): (ROOT / "protocol/eval_selected.csv", "f5w_p005_"),
    (7, 0.003): (ROOT / "protocol/eval_selected.csv", "f7w_p003_"),
    (7, 0.004): (SWEEP / "gru_test_f7w_p004_eval_best_ckpt.csv", "f7w_p004_"),
    (7, 0.005): (ROOT / "protocol/eval_selected.csv", "f7w_p005_"),
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


def fit_collapse(sweep, window=COLLAPSE_WINDOW):
    # P_L = a + b x + c x^2, x = (p - p_th) d^(1/nu), weighted by binomial errors; errors on
    # (p_th, nu) from the covariance scaled by sqrt(chi2/dof), since the quadratic ansatz is not exact
    rows = [(d, p, v["errors"] / v["shots"], v["shots"]) for (d, p), v in sweep.items()
            if window[0] <= p <= window[1]]
    d, p, y, n = map(np.array, zip(*rows))
    sigma = np.sqrt(y * (1 - y) / n)
    model = lambda t, d, p: np.polyval(t[:1:-1], (p - t[0]) * d ** (1 / t[1]))
    fit = least_squares(lambda t: (model(t, d, p) - y) / sigma, [0.0072, 1.0, 0.03, 10.0, 100.0])
    chi2, dof = float((fit.fun ** 2).sum()), len(y) - len(fit.x)
    err = np.sqrt(np.diag(np.linalg.inv(fit.jac.T @ fit.jac)) * chi2 / dof)
    return dict(p_th=fit.x[0], p_th_err=err[0], nu=fit.x[1], nu_err=err[1], coef=list(fit.x[2:]),
                chi2=chi2, dof=dof, window=window, points=len(y),
                x=(p - fit.x[0]) * d ** (1 / fit.x[1]), y=y, d=d)


def add_collapse_inset(fig, c):
    # 4.5 x 2.6 in frame (tick and axis labels inside it) in the empty lower right of the main axes
    w, h = fig.get_size_inches()
    ax = fig.add_axes([(9.3 + 0.62) / w, (0.86 + 0.62) / h, 3.78 / w, 1.88 / h])
    for d in sorted(set(c["d"])):
        m = c["d"] == d
        ax.plot(c["x"][m], c["y"][m], "o", ms=4, color=COLORS[d], zorder=2)
    xs = np.linspace(c["x"].min(), c["x"].max(), 200)
    ax.plot(xs, np.polyval(c["coef"][::-1], xs), color="0.35", lw=1, zorder=1)
    # Helvetica's nu looks like an italic v; \mathsf takes just that glyph from DejaVu Sans
    plt.rcParams["mathtext.sf"] = "DejaVu Sans:italic"
    ax.text(0.04, 0.95, f"$\\mathsf{{\\nu}}$ = {c['nu']:.2f} $\\pm$ {c['nu_err']:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=poster_style.TICK_PT)
    # x = 0 is p = p_th; one significant figure since the fit moves p_th over 0.69-0.76% across
    # windows and minimum d
    ax.axvline(0, color=poster_style.NEUTRAL, ls=(0, (4, 3)), lw=0.7, zorder=0)
    ax.annotate(f"$p = p_{{\\mathrm{{th}}}} \\approx$ {100 * c['p_th']:.1f}%", xy=(0, 0.05),
                xycoords=ax.get_xaxis_transform(), xytext=(4, 0), textcoords="offset points",
                ha="left", va="bottom", fontsize=poster_style.TICK_PT - 2, color=poster_style.NEUTRAL)
    ax.set_xlabel("Scaled error rate $(p - p_{\\mathrm{th}})\\,d^{\\,1/\\mathsf{\\nu}}$",
                  fontsize=poster_style.TICK_PT, labelpad=2)
    ax.set_ylabel("$P_L$ (MWPM)", fontsize=poster_style.TICK_PT, labelpad=2)
    ax.set_title("MWPM threshold estimate", fontsize=poster_style.TICK_PT, pad=4)
    ax.tick_params(labelsize=poster_style.TICK_PT)
    ax.locator_params(axis="both", nbins=4)
    return ax


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


def add_improvement_note(ax, stars):
    # improvement against MWPM decoded on the same test shots as each GRU star, not against the sweep curve
    gain = [100 * (1 - s["gru"] / s["mwpm"]) for s in stars]
    target = max((s for s in stars if s["p"] == OPERATING_P), key=lambda s: s["gru"])
    ax.annotate(f"GRU achieves {min(gain):.0f}%-{max(gain):.0f}% lower\nlogical-error rates than MWPM",
                xy=(target["p"], target["gru"]), xytext=(0.00135, 4e-2), ha="left", va="center",
                fontsize=poster_style.TICK_PT, color=poster_style.INK,
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=poster_style.INK, shrinkA=4, shrinkB=15,
                                mutation_scale=10))
    ax.text(0.012, 0.70, "Lower is better \u2193", transform=ax.transAxes, ha="left", va="top",
            fontsize=poster_style.TICK_PT, color=poster_style.NEUTRAL)
    return gain


def plot(points, stars, a_fit, p_th):
    fig, ax = plt.subplots(figsize=(14.25, 6.0))
    ax.axvline(OPERATING_P, color="0.6", lw=0.8, zorder=0)
    for d in sorted({d for d, _ in points}):
        ps = sorted(p for dd, p in points if dd == d)
        v = [points[d, p] for p in ps]
        pl = [x["errors"] / x["shots"] for x in v]
        band = np.array([likelihood_band(x["errors"], x["shots"]) for x in v])
        c = COLORS[d]
        ax.fill_between(ps, band[:, 0], band[:, 1], color=c, alpha=0.15, lw=0, zorder=1)
        ax.plot(ps, pl, color=c, ls=STYLES[d], lw=2, marker="o", ms=4, mfc=c, mec=c, zorder=2)
        ax.annotate(f"$d$ = {d}", (ps[0], pl[0]), textcoords="offset points", xytext=(-8, 0),
                    ha="right", va="center", color=c, fontsize=poster_style.CURVE_LABEL_PT)
        # guide only over this d's own measured range, and only below threshold
        p_guide = np.geomspace(min(ps), min(max(ps), p_th), 100)
        ax.plot(p_guide, a_fit * (p_guide / p_th) ** ceil(d / 2), color=c, ls=(0, (5, 4)), lw=1,
                zorder=1)
    for s in stars:
        ax.plot(s["p"], s["gru"], marker="*", ms=26, color=COLORS[s["d"]], mec=STAR_EDGE, mew=1,
                ls="none", zorder=4)

    handles = [plt.Line2D([], [], color=poster_style.NEUTRAL, lw=2),
               plt.Line2D([], [], marker="*", ms=26, color="0.35", mec=STAR_EDGE, mew=1, ls="none"),
                plt.Line2D([], [], color="0.35", ls=(0, (5, 4)), lw=1)]
    ax.legend(handles, ["MWPM", "GRU", "Fit"], loc="upper left", frameon=False, handlelength=2.6)
    add_improvement_note(ax, stars)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(7.2e-4, 1.65e-2)  # room for the curve labels left of p = 0.001
    x_ticks = [0.001, 0.002, 0.004, 0.008, 0.015]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{t:g}" for t in x_ticks])
    ax.set_xlabel("Physical error rate $p$")
    ax.set_ylabel("Logical error rate $P_L$")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    sweep = load_sweep(SWEEP / "sinter_stats.csv")
    points = {k: v for k, v in sweep.items() if v["errors"] >= MIN_ERRORS}
    skipped = sorted(k for k in sweep if k not in points)
    a_fit, p_th, n_fit = fit_threshold(points)
    stars = load_gru()

    poster_style.apply()
    fig = plot(points, stars, a_fit, p_th)
    collapse = fit_collapse(sweep)
    add_collapse_inset(fig, collapse)
    POSTER.mkdir(parents=True, exist_ok=True)
    fig.savefig(POSTER / "fig_threshold.png", dpi=300, facecolor="white")
    fig.savefig(POSTER / "fig_threshold_transparent.png", dpi=300, transparent=True)
    fig.savefig(POSTER / "fig_threshold.svg", facecolor="white")

    summary = dict(p_th=p_th, A=a_fit, points_in_fit=n_fit, skipped_below_100_errors=skipped,
                   total_shots=sum(v["shots"] for v in sweep.values()),
                   total_cpu_s=sum(v["cpu_s"] for v in sweep.values()),
                   collapse={k: v for k, v in collapse.items() if k not in ("x", "y", "d")},
                   gru=[dict(s, ratio=s["gru"] / s["mwpm"]) for s in stars])
    json.dump(summary, open(OUT / "fig_threshold_summary.json", "w"), indent=1, default=list)
    with open(OUT / "fig_threshold_points.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["d", "p", "shots", "errors", "P_L", "cpu_s", "plotted", "in_fit"])
        for (d, p), v in sorted(sweep.items()):
            w.writerow([d, p, v["shots"], v["errors"], f"{v['errors'] / v['shots']:.4e}",
                        round(v["cpu_s"], 1), (d, p) in points, (d, p) in points and p < p_th])
    print(f"p_th = {p_th:.5f}  A = {a_fit:.4f}  ({n_fit} points in fit); skipped {skipped}")
    print(f"collapse, p in {collapse['window']} ({collapse['points']} points): "
          f"p_th = {collapse['p_th']:.5f} +- {collapse['p_th_err']:.5f}, nu = {collapse['nu']:.3f} "
          f"+- {collapse['nu_err']:.3f}, chi2/dof = {collapse['chi2']:.1f}/{collapse['dof']}")
    for s in stars:
        print(f"d={s['d']} p={s['p']}: GRU {s['gru']:.6f} MWPM {s['mwpm']:.6f} "
              f"ratio {s['gru'] / s['mwpm']:.3f}  {s['source']}  {s['weights']}")
