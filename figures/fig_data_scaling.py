# Created: 2026-10-28
# Last modified: 2026-10-28 (running-minimum frontier per d, faded non-frontier runs, 8B marker jitter)
# GRU / MWPM ratio at p = 0.004 against unique training shots: dataset size for fixed sets,
# 100M x epochs for fresh-shot runs. Hollow markers: older runs checkpointed by min val loss.
#   python figures/fig_data_scaling.py
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter

import poster_style

ROOT = Path(__file__).resolve().parents[1]
POSTER = ROOT / "figures" / "poster"
FRAME_IN = (6.9, 4.6)
LABEL_PT, TICK_PT = 13, 11
MARKER_MS = 6
OFF_FRONTIER_SCALE, OFF_FRONTIER_ALPHA = 0.7, 0.35
# d = 5 and d = 7 fresh-shot runs both sit at 8B; markers only are shifted by this many decades
X_JITTER_DEX = {(5, 8e9): -0.03, (7, 8e9): 0.03}
DEPTH_MARKER = {1: "o", 2: "s", 3: "^"}
# d = 9 point labels as (text, offset in points, ha); "fresh" goes on the best fresh-shot run
LABELLED_D9 = {1e7: ("10M", (-MARKER_MS, 0), "right"), 1e8: ("100M", (MARKER_MS, MARKER_MS + 2), "left")}
FRESH_LABEL = ("fresh", (0, MARKER_MS + 2), "center")


def unique_shots(r):
    if r["Data"].startswith("fresh"):
        return 1e8 * int(r["Max Epoch"])
    return float(re.fullmatch(r"(\d+)M", r["Data"])[1]) * 1e6


def excluded(r):
    if r["Done?"] != "Yes" or r["p"] != "0.004" or r["reset_after"] == "TRUE":
        return "not a completed p = 0.004 reset_after=False run"
    for word in ("distilled", "diverged", "EAF"):
        if word in r["Notes"]:
            return word
    return None


def load():
    path = sorted(ROOT.glob("GRU_all_runs_*.csv"))[-1]
    runs, skipped = [], []
    for r in csv.DictReader(open(path)):
        reason = excluded(r)
        if reason:
            skipped.append((reason, r))
            continue
        width = r["Width"]
        runs.append(dict(d=int(r["d"]), depth=int(width[0]) if "x" in width else 1, width=width,
                         shots=unique_shots(r), ratio=float(r["p_L"]) / float(r["MWPM"]),
                         min_val_loss=r["Checkpoint criterion"].startswith("min val loss"),
                         data=r["Data"], notes=r["Notes"][:50]))
    return path, runs, skipped


def frontier(runs, d):
    # running minimum: a run is on the frontier if it beats every run with fewer or equal unique shots
    points, lowest = [], float("inf")
    for r in sorted((r for r in runs if r["d"] == d), key=lambda r: (r["shots"], r["ratio"])):
        if r["ratio"] < lowest and not (points and points[-1]["shots"] == r["shots"]):
            points.append(r)
            lowest = r["ratio"]
    return points


def plot(runs):
    fig, ax = plt.subplots(figsize=FRAME_IN)
    for d in sorted({r["d"] for r in runs}):
        color = poster_style.DIST[d]
        front = frontier(runs, d)
        ax.plot([r["shots"] for r in front], [r["ratio"] for r in front], lw=1, color=color, zorder=2)
        for r in (r for r in runs if r["d"] == d):
            on_front = any(r is f for f in front)
            x = r["shots"] * 10 ** X_JITTER_DEX.get((d, r["shots"]), 0)
            ax.plot(x, r["ratio"], ls="none", marker=DEPTH_MARKER[r["depth"]],
                    ms=MARKER_MS if on_front else MARKER_MS * OFF_FRONTIER_SCALE,
                    alpha=1 if on_front else OFF_FRONTIER_ALPHA, mec=color,
                    mfc="white" if r["min_val_loss"] else color, mew=1.2, zorder=4 if on_front else 3)
        if d == 9:
            fresh = min((r for r in front if r["data"].startswith("fresh")), key=lambda r: r["ratio"])
            for r in front:
                label = LABELLED_D9.get(r["shots"], FRESH_LABEL if r is fresh else None)
                if label:
                    text, offset, ha = label
                    ax.annotate(text, (r["shots"], r["ratio"]), xytext=offset, textcoords="offset points",
                                ha=ha, va="center", color=color, fontsize=TICK_PT)

    ax.axhline(1.0, color=poster_style.INK, lw=0.8, ls="--", zorder=1)
    ax.annotate("MWPM", (0, 1.0), xycoords=ax.get_yaxis_transform(), xytext=(3, 1),
                textcoords="offset points", ha="left", va="bottom", fontsize=TICK_PT, color=poster_style.INK)

    ink = poster_style.INK
    legend_kw = dict(frameon=False, fontsize=TICK_PT, handletextpad=0.4, borderaxespad=0.3)
    # distance key sits in the empty corner under the MWPM line
    d_key = ax.legend([Line2D([], [], color=poster_style.DIST[d], lw=1, marker="o", ms=MARKER_MS) for d in (5, 7, 9)],
                      [f"$d$ = {d}" for d in (5, 7, 9)], loc="lower left", handlelength=1.6, **legend_kw)
    ax.add_artist(d_key)
    marker_handles = [Line2D([], [], ls="none", color=ink, marker=DEPTH_MARKER[n], ms=MARKER_MS) for n in (1, 2, 3)]
    marker_handles.append(Line2D([], [], ls="none", color=ink, marker="o", mfc="white", mew=1.2, ms=MARKER_MS))
    ax.legend(marker_handles, ["1 GRU layer", "2 GRU layers", "3 GRU layers", "hollow: checkpoint by min val loss"],
              loc="upper right", handlelength=1.0, **legend_kw)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.2e6, 4e10)
    ax.set_ylim(0.5, 25)
    ax.set_xticks([1e6, 1e7, 1e8, 1e9, 1e10])
    ax.set_xticklabels(["1M", "10M", "100M", "1B", "10B"])
    y_ticks = [0.5, 1, 2, 5, 10, 20]
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.set_yticklabels([f"{t:g}" for t in y_ticks])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Unique training shots")
    ax.set_ylabel("Error ratio (GRU / MWPM)")
    fig.tight_layout(pad=0.3)
    return fig


if __name__ == "__main__":
    path, runs, skipped = load()
    poster_style.apply()
    plt.rcParams.update({"axes.labelsize": LABEL_PT, "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT})
    fig = plot(runs)
    POSTER.mkdir(parents=True, exist_ok=True)
    fig.savefig(POSTER / "fig_data_scaling.png", dpi=300, facecolor="white")
    fig.savefig(POSTER / "fig_data_scaling_transparent.png", dpi=300, transparent=True)

    print(path.name, f"{len(runs)} runs plotted, {len(skipped)} skipped")
    for r in sorted(runs, key=lambda r: (r["d"], r["shots"], r["ratio"])):
        print(f"d={r['d']} {r['width']:7s} {r['data']:17s} shots {r['shots']:.1e}  ratio {r['ratio']:.3f}"
              f"{'  min val loss' if r['min_val_loss'] else ''}")
    for d in (5, 7, 9):
        print(f"frontier d={d}:", ", ".join(f"{r['shots']:.1e} {r['width']} {r['ratio']:.3f}" for r in frontier(runs, d)))
    for reason, r in skipped:
        if r["p"] == "0.004" and r["Done?"] == "Yes" and r["reset_after"] == "FALSE":
            print(f"skipped ({reason}): d={r['d']} {r['Width']} {r['Data']} {r['Relative to MWPM']}")
