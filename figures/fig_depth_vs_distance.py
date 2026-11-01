# Created: 2026-10-28
# Last modified: 2026-10-28
# Depth vs distance: (a) GRU / MWPM on the eval block vs number of GRU layers, one line per d;
# (b) L_min, the fewest layers that beat MWPM (ratio < 1, McNemar p < 0.01), vs d.
#   python figures/fig_depth_vs_distance.py --root experiments/depth_vs_distance
import argparse
import csv
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poster_style as ps

DISTANCES = (5, 7, 9)
MARKERS = {5: "o", 7: "s", 9: "D"}   # second cue beside colour: DIST 7 and 9 are close in hue

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", default=None)
args = ap.parse_args()
out_base = args.out or os.path.join(args.root, "fig_depth_vs_distance")

ratio = {}   # (d, L, seed) -> GRU / MWPM
for r in csv.DictReader(open(os.path.join(args.root, "GRU_depth_vs_distance_runs.csv"))):
    m = re.fullmatch(r"dvd_d(\d)_L(\d\d)_u\d+(_s1)?", r["Run"])
    if m and r["Relative to MWPM"]:
        ratio[(int(m.group(1)), int(m.group(2)), 1 if m.group(3) else 0)] = float(r["Relative to MWPM"])
lmin = {}
for d in DISTANCES:
    f = os.path.join(args.root, "lmin", f"d{d}.json")
    if os.path.exists(f):
        lmin[d] = json.load(open(f)).get("lmin")

ps.apply()
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.6, 1]})

ax_a.axhline(1, color=ps.NEUTRAL, lw=2, ls="--", zorder=1)
ax_a.annotate("MWPM", (5.9, 1), xytext=(0, 4), textcoords="offset points", color=ps.NEUTRAL,
              va="bottom", ha="right", fontsize=ps.CURVE_LABEL_PT)
for d in DISTANCES:
    pts = sorted((L, v) for (dd, L, s), v in ratio.items() if dd == d and s == 0)
    if not pts:
        continue
    Ls, vs = zip(*pts)
    ax_a.plot(Ls, vs, color=ps.DIST[d], lw=2, marker=MARKERS[d], ms=8, mec="white", mew=1.5,
              label=f"d = {d}", zorder=3)
    s1 = sorted((L, v) for (dd, L, s), v in ratio.items() if dd == d and s == 1)
    if s1:
        ax_a.plot(*zip(*s1), ls="none", marker=MARKERS[d], ms=9, mfc="white", mec=ps.DIST[d], mew=2, zorder=4)
ax_a.set_xlabel("GRU layers")
ax_a.set_ylabel("GRU / MWPM logical error rate")
ax_a.set_xticks(range(1, 6))
ax_a.set_xlim(0.7, 5.9)
ax_a.text(0, 1.03, "(a)", transform=ax_a.transAxes, va="bottom", fontsize=ps.AXIS_LABEL_PT)
if any(s == 1 for (_, _, s) in ratio):
    ax_a.plot([], [], ls="none", marker="o", mfc="white", mec=ps.INK, mew=2, ms=9, label="second seed")
ax_a.legend(frameon=False, loc="upper right")

ds = [d for d in DISTANCES if lmin.get(d)]
for d in ds:
    ax_b.plot(d, lmin[d], ls="none", marker=MARKERS[d], ms=12, color=ps.DIST[d], mec="white", mew=1.5, zorder=3)
if len(ds) > 1:
    ax_b.plot(ds, [lmin[d] for d in ds], color=ps.INK, lw=1, zorder=2)
for d in DISTANCES:
    if d in lmin and not lmin[d]:
        ax_b.text(d, 5.4, "none\nin 1-5", ha="center", va="top", color=ps.INK, fontsize=ps.TICK_PT)
ax_b.set_xlabel("Code distance d")
ax_b.set_ylabel("Fewest layers that beat MWPM")
ax_b.set_xticks(DISTANCES)
ax_b.set_yticks(range(1, 6))
ax_b.set_xlim(4, 10)
ax_b.set_ylim(0.5, 5.5)
ax_b.text(0, 1.03, "(b)", transform=ax_b.transAxes, va="bottom", fontsize=ps.AXIS_LABEL_PT)

for ax in (ax_a, ax_b):
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{out_base}.{ext}", dpi=300)
print(f"wrote {out_base}.png/.pdf")
