# Created: 2026-10-20
# Last modified: 2026-10-20

"""Distillation pilot figure from the newest GRU_distillation_runs_*.csv.

Only finished pilot rows and their teachers are plotted.

python figures/distillation_fig.py      -> figures/output/distillation_pilot.{pdf,png}
"""
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scaling_figs import N_EVAL, OUT_DIR, REPO, wilson  # also applies the shared rcParams

D_COLOR = {5: "#0072B2", 7: "#D55E00", 9: "#009E73"}
SEED_NOISE = 0.02  # spread of same-recipe d9 teacher seeds on the eval block, x MWPM


def load():
    path = sorted(REPO.glob("GRU_distillation_runs_*.csv"))[-1]
    rows = [r for r in csv.DictReader(open(path)) if r["Done?"] == "Yes"]
    teachers = {int(r["d"]): r for r in rows if r["Role"] == "teacher" and not r["Notes"].startswith("earlier probe")}
    students = [r for r in rows if r["Role"] == "student" and r["Notes"].startswith("kd_pilot")]
    return path, teachers, students


def ratio_ci(r):
    lo, hi = wilson(float(r["p_L"]) * N_EVAL, N_EVAL)
    return float(r["Relative to MWPM"]), lo / float(r["MWPM"]), hi / float(r["MWPM"])


def mcnemar_bc(s):
    b, c = re.match(r"b=(\d+) c=(\d+)", s).groups()
    return int(b), int(c)


def plot(teachers, students):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.2))

    for d, t in sorted(teachers.items()):
        y, lo, hi = ratio_ci(t)
        ax_a.errorbar(int(t["Parameters"]), y, yerr=[[y - lo], [hi - y]], fmt="*", ms=13,
                      color=D_COLOR[d], mec="black", mew=0.6, zorder=4)
    for r in students:
        d, n = int(r["d"]), int(r["Parameters"])
        y, lo, hi = ratio_ci(r)
        distilled = r["alpha"] == "0"
        one_layer = "x" not in r["Width"] and d > 5
        long_run = r["Run"].endswith("_e160")
        marker = "X" if one_layer else ("D" if long_run else "o")
        x = n * (0.95 if distilled else 1.05) if not one_layer else n  # separate the two arms of a pair
        ax_a.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=marker, ms=6.5, color=D_COLOR[d],
                      mfc=D_COLOR[d] if distilled else "white", mew=1.4, zorder=3)
    ax_a.axhline(1.0, color="black", lw=1, ls="--")
    ax_a.text(1.6e4, 1.03, "MWPM", fontsize=8)
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("parameters")
    ax_a.set_ylabel("logical error rate / MWPM (eval block)")
    ax_a.set_title("Students and teachers, p = 0.004")
    ax_a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_a.yaxis.set_minor_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}" if v in (0.7, 0.8, 1.5, 2, 3, 5) else ""))

    handles = [plt.Line2D([], [], ls="", marker="o", color=D_COLOR[d], ms=7, label=f"d = {d}") for d in sorted(teachers)]
    handles += [plt.Line2D([], [], ls="", marker="*", color="grey", mec="black", ms=11, label="teacher"),
                plt.Line2D([], [], ls="", marker="o", color="grey", ms=6.5, label="distilled"),
                plt.Line2D([], [], ls="", marker="o", color="grey", mfc="white", mew=1.4, ms=6.5, label="hard label"),
                plt.Line2D([], [], ls="", marker="D", color="grey", ms=6, label="160 epochs (d = 5 u70)"),
                plt.Line2D([], [], ls="", marker="X", color="grey", ms=7, label="single layer, distilled")]
    ax_a.legend(handles=handles, loc="upper right")

    pairs = [r for r in students if r["alpha"] == "0" and r["McNemar vs hard label"]]
    ax_b.axhspan(-SEED_NOISE, SEED_NOISE, color="#E6E6E6", zorder=0)
    ax_b.axhline(0, color="black", lw=1)
    for r in pairs:
        d = int(r["d"])
        b, c = mcnemar_bc(r["McNemar vs hard label"])
        mwpm_errors = float(r["MWPM"]) * N_EVAL
        gain = (b - c) / mwpm_errors
        err = 1.959964 * np.sqrt(b + c) / mwpm_errors
        comp = float(r["Compression"].rstrip("x"))
        long_run = r["Run"].endswith("_e160")
        ax_b.errorbar(comp, gain, yerr=err, fmt="D" if long_run else "o", ms=7, color=D_COLOR[d], mew=1.4, zorder=3)
        ax_b.annotate(r["Width"] + (", 160 ep" if long_run else ""), (comp, gain), xytext=(-8, 6) if long_run else (8, -3),
                      textcoords="offset points", fontsize=8, ha="right" if long_run else "left")
    ax_b.text(1.02, SEED_NOISE, "seed-to-seed\nspread", transform=ax_b.get_yaxis_transform(), fontsize=7,
              va="center", color="#555555")
    ax_b.set_xscale("log")
    ax_b.set_xlim(2.5, 25)
    ax_b.set_xticks([3, 4, 6, 10, 15, 20])
    ax_b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax_b.xaxis.set_minor_formatter(plt.NullFormatter())
    ax_b.set_xlabel("compression (teacher params / student params)")
    ax_b.set_ylabel("hard label - distilled (x MWPM)")
    ax_b.set_title("Gain from distillation")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    path, teachers, students = load()
    fig = plot(teachers, students)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "distillation_pilot.pdf")
    fig.savefig(OUT_DIR / "distillation_pilot.png", dpi=300)
    print(f"{path.name}: {len(students)} students -> {OUT_DIR / 'distillation_pilot.png'}")
