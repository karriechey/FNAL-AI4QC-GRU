# Created: 2026-10-18
# Last modified: 2026-10-18 (standard wording on all labels)

"""GRU vs MWPM at p = 0.003, 0.004, 0.005 for d = 5, 7, 9 on the evaluation block.

Checkpoints chosen by protocol/selection_rule.md.

python figures/error_rate_scaling.py [--csv GRU_all_runs_2026-10-01.csv]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "output"
N_EVAL = 1_800_000

HEADLINE = {
    (5, 0.003): "f5w_p003", (5, 0.004): "f5w_p004", (5, 0.005): "f5w_p005",
    (7, 0.003): "f7w_p003", (7, 0.004): "f7w_p004", (7, 0.005): "f7w_p005",
    (9, 0.003): "f9w_p003", (9, 0.004): "f3280e240", (9, 0.005): "f9w_p005",
}
COLORS = {5: "tab:blue", 7: "tab:orange", 9: "tab:purple"}
MWPM_COLOR = "tab:green"
# ratio label offsets in points, per (d, p), so neighbouring labels do not overlap
LABEL_OFFSET = {(5, 0.003): (0, 7), (5, 0.004): (0, 7), (5, 0.005): (0, 7),
                (7, 0.003): (0, -13), (7, 0.004): (8, 2), (7, 0.005): (8, 2),
                (9, 0.003): (8, -4), (9, 0.004): (8, -11), (9, 0.005): (8, -11)}


def wilson(p_hat, n, conf=0.95):
    z = norm.ppf(0.5 + conf / 2)
    centre = (p_hat + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def load(csv_path):
    runs = pd.read_csv(csv_path)
    rows = []
    for (d, p), tag in HEADLINE.items():
        hit = runs[runs["Notes"].fillna("").str.startswith(f"probe {tag};")]
        if len(hit) != 1:
            raise SystemExit(f"{tag}: {len(hit)} rows in {csv_path}")
        r = hit.iloc[0]
        rows.append(dict(d=d, p=p, run=tag, epochs=int(r["Max Epoch"]),
                         gru=float(r["p_L"]), mwpm=float(r["MWPM"])))
    return pd.DataFrame(rows)


def plot(df):
    fig, (ax_pl, ax_ratio) = plt.subplots(1, 2, figsize=(10, 4.2))
    for d, g in df.groupby("d"):
        g = g.sort_values("p")
        c = COLORS[d]
        lo, hi = wilson(g["gru"].values, N_EVAL)
        ax_pl.errorbar(g["p"], g["gru"], yerr=[g["gru"] - lo, hi - g["gru"]], color=c,
                       marker="o", capsize=3, label=f"GRU d={d}")
        ax_pl.plot(g["p"], g["mwpm"], color=MWPM_COLOR, marker="s", ls=":", mfc="none",
                   label="MWPM" if d == 5 else None)
        ax_pl.annotate(f"d={d}", (g["p"].iloc[0], g["mwpm"].iloc[0]), textcoords="offset points",
                       xytext=(-6, 0), ha="right", va="center", fontsize=8, color=MWPM_COLOR)
        ax_ratio.plot(g["p"], g["gru"] / g["mwpm"], color=c, marker="o", label=f"d={d}")
        for p, ratio in zip(g["p"], g["gru"] / g["mwpm"]):
            ax_ratio.annotate(f"{ratio:.3f}", (p, ratio), textcoords="offset points",
                              xytext=LABEL_OFFSET[d, p], ha="center" if LABEL_OFFSET[d, p][0] == 0 else "left",
                              fontsize=8, color=c)

    ticks = sorted(df["p"].unique())
    ax_pl.set_xscale("log")
    ax_pl.set_yscale("log")
    ax_pl.set_xlabel("physical error rate p")
    ax_pl.set_ylabel("logical error rate per shot (d QEC rounds)")
    ax_pl.set_title("Logical vs physical error rate")
    y_ticks = [0.0004, 0.0006, 0.001, 0.002, 0.004, 0.006, 0.01, 0.015]
    ax_pl.yaxis.set_major_locator(plt.FixedLocator(y_ticks))
    ax_pl.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_pl.yaxis.set_minor_locator(plt.NullLocator())
    ax_pl.set_ylim(0.0004, 0.016)
    ax_pl.legend(fontsize=8)
    ax_pl.set_xlim(0.0027, 0.0053)

    ax_ratio.axhline(1.0, color="gray", lw=0.8)
    ax_ratio.set_xlabel("physical error rate p")
    ax_ratio.set_ylabel("logical error rate ratio, GRU / MWPM")
    ax_ratio.set_ylim(0.5, 1.0)
    ax_ratio.set_title("GRU relative to MWPM (below 1 = GRU better)")
    ax_ratio.text(0.02, 0.02, "d=9, p=0.003: training not converged after 240 epochs", transform=ax_ratio.transAxes,
                  fontsize=8, color=COLORS[9])
    ax_ratio.legend(fontsize=8)

    for ax in (ax_pl, ax_ratio):
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.3f}" for t in ticks])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=ROOT / "GRU_all_runs_2026-10-01.csv")
    args = ap.parse_args()
    df = load(args.csv)
    OUT.mkdir(parents=True, exist_ok=True)
    df.assign(ratio=(df["gru"] / df["mwpm"]).round(4)).to_csv(OUT / "error_rate_scaling.csv", index=False)
    fig = plot(df)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"error_rate_scaling.{ext}", dpi=300)
    print(df.assign(ratio=(df["gru"] / df["mwpm"]).round(3)).to_string(index=False))
