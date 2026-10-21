# Created: 2026-10-05
# Last updated: 2026-10-06 (distilled runs are their own recipe; summary_md writes each notebook section's text from its figure)

"""
Scaling-law and comparison figures for the GRU decoder study.

Shared by notebooks/scaling_laws.ipynb and the command line. Writes to figures/output/:
a PDF and 300-dpi PNG per figure, scaling_summary.pdf, fit_parameters.csv/.md,
lambda_table.csv and runs_used.csv.

Filled markers are evaluation-block scores only; pending runs (figures/data/pending_runs.csv)
are hollow, validation p_L only. Each fitted series is one recipe (check_one_recipe).
Diverged, unstable or killed runs are drawn as "x" and never fitted. Error bars are Wilson 95%.

python figures/scaling_figs.py                  # newest GRU_all_runs_*.csv in the repo root
python figures/scaling_figs.py --csv path.csv   # a specific all-runs table
"""

from __future__ import annotations

import argparse
import glob
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg") if __name__ == "__main__" else None
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
from scipy.optimize import curve_fit
from scipy.stats import norm

# --------------------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "figures" / "data"
OUT_DIR = REPO / "figures" / "output"

# evaluation block [15.2M, 17M); p=0.003 and 0.005 FORMAL pools use the same partition
N_EVAL = 1_800_000
N_VAL = 200_000  # validation block [15.0M, 15.2M)

N_BOOT = 2000  # bootstrap replicates for every fit and every Lambda
RNG_SEED = 20261005  # fixed so reruns reproduce the same CIs

# Okabe-Ito colours, one per width, same in every figure (yellow left out)
WIDTH_COLOR = {
    "u70": "#E69F00",     # orange
    "u140": "#56B4E9",    # sky blue
    "u200": "#009E73",    # bluish green
    "u280": "#0072B2",    # blue
    "u560": "#D55E00",    # vermillion
    "2xu280": "#CC79A7",  # reddish purple
    "2xu160": "#882255",  # wine (stacked GRUs within the u280 budget)
    "2xu140": "#AA4499",  # purple
}
MWPM_COLOR = "#000000"
NEUTRAL = "#7F7F7F"  # used for references (best-of envelope, train-slice markers)

# one marker per code distance
D_MARKER = {5: "o", 7: "s", 9: "^"}

# eval-block MWPM per distance at p=0.004 (re-read from the CSV at load time)
MWPM_FALLBACK = {5: 0.007571, 7: 0.004291, 9: 0.002267}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "lines.linewidth": 1.6,
    "errorbar.capsize": 0,
    "pdf.fonttype": 42,  # embed TrueType so PDF text stays editable
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})


# --------------------------------------------------------------------------------------
# Loading and labelling runs
# --------------------------------------------------------------------------------------
def newest_runs_csv() -> Path:
    # newest GRU_all_runs_<date>.csv in the repo root
    hits = sorted(REPO.glob("GRU_all_runs_*.csv"))
    if not hits:
        raise FileNotFoundError(f"no GRU_all_runs_*.csv in {REPO}")
    return hits[-1]


def parse_shots(s) -> float:
    # '100M' -> 1e8, '2M' -> 2e6
    s = str(s).strip()
    m = re.fullmatch(r"([\d.]+)\s*([kKmM]?)", s)
    if not m:
        return np.nan
    mult = {"": 1, "k": 1e3, "K": 1e3, "m": 1e6, "M": 1e6}[m.group(2)]
    return float(m.group(1)) * mult


def fmt_shots(n: float) -> str:
    if n >= 1e6:
        v = n / 1e6
        return f"{v:g}M"
    return f"{n / 1e3:g}k"


def fmt_params(p: float) -> str:
    if p >= 1e6:
        return f"{p / 1e6:g}M"
    return f"{p / 1e3:g}k"


def fmt_lr(lr: float) -> str:
    # 0.003 -> '3e-3', 0.0015 -> '1.5e-3'
    m, e = f"{lr:.2e}".split("e")
    return f"{float(m):g}e{int(e)}"


def load_runs(csv_path: Path | str | None = None) -> pd.DataFrame:
    # Read the all-runs CSV and parse the free-text 'Checkpoint criterion' column into
    # explicit recipe columns (pipeline, warmup, data pool, divergence).
    csv_path = Path(csv_path) if csv_path else newest_runs_csv()
    raw = pd.read_csv(csv_path)
    df = pd.DataFrame({
        "csv_line": raw.index + 2,  # line number in the CSV file (header is line 1)
        "status_raw": raw["Done?"].astype(str),
        "width": raw["Width"].astype(str),
        "params": raw["Parameters"].astype(float),
        "d": raw["d"].astype(int),
        "schedule_raw": raw["LR Schedule"].astype(str),
        "lr": raw["LR"].astype(float),
        "data": raw["Data"].astype(str),
        "seed": raw["Seed"].astype(int),
        "p": raw["p"].astype(float),
        "batch": raw["Batch"].astype(int),
        "epochs": raw["Max Epoch"].astype(int),
        "best_epoch": raw["Best Ckpt"],
        "mwpm": raw["MWPM"].astype(float),
        "p_L": raw["p_L"].astype(float),
        "reset_after": raw["reset_after"].astype(str).str.upper().eq("TRUE"),
        "note": raw["Checkpoint criterion"].astype(str),
        "date": raw["Date"].astype(str),
    })
    note = df["note"].str.lower()
    df["N"] = df["data"].map(parse_shots)
    df["schedule"] = np.where(df["schedule_raw"].str.startswith("cosine"), "cosine", "constant")
    df["ckpt"] = np.where(note.str.contains("min val p_l"), "val p_L",
                          np.where(note.str.contains("min val loss"), "val loss", "other"))
    df["pipeline"] = np.where(note.str.contains("older pipeline|older run"), "old", "new")
    df["pool"] = np.where(note.str.contains("eaf"), "EAF", "grace1")
    df["warmup"] = note.str.contains("warmup")
    df["warm_start"] = note.str.contains("warm start")
    # a distilled student is its own recipe, never a replicate of the label-trained model
    df["distilled"] = note.str.contains("distilled")
    df["diagnostic"] = df["status_raw"].str.contains("diagnostic") | note.str.contains("overfit test")
    df["stability"] = np.select(
        [df["status_raw"].str.startswith("Killed"), note.str.contains("diverged"),
         note.str.contains("unstable")],
        ["killed", "diverged", "unstable"], default="stable")
    # scored: finished, has p_L, not a diagnostic, FORMAL pool on grace1
    df["scored"] = (df["status_raw"].str.startswith("Yes") & df["p_L"].notna()
                    & ~df["diagnostic"] & df["pool"].eq("grace1"))
    df["n_shots"] = N_EVAL
    df["k"] = np.round(df["p_L"] * df["n_shots"])
    df["ratio"] = df["p_L"] / df["mwpm"]
    df["val_only"] = False
    df["recipe"] = df.apply(recipe_string, axis=1)
    df["label"] = df.apply(
        lambda r: f"d{r.d} {r.width} {r.data} s{r.seed} {r.schedule} {fmt_lr(r.lr)} {r.epochs}ep"
                  + (" distilled" if r.distilled else ""), axis=1)
    df.attrs["source"] = csv_path.name
    return df


def recipe_string(r) -> str:
    # everything except d, width, data size and seed
    bits = [r.schedule, f"peak {fmt_lr(r.lr)}", f"{r.epochs} ep", f"batch {r.batch}",
            f"ckpt {r['ckpt']}", f"{r.pipeline} pipeline", f"reset_after={r.reset_after}",
            "warmup 3 ep" if r.warmup else "no warmup", f"p={r.p:g}", f"{r.pool} pool"]
    if r.warm_start:
        bits.append("warm start")
    if r.get("distilled", False):
        bits.append("distilled")
    return ", ".join(bits)


RECIPE_KEYS = ["schedule", "lr", "epochs", "batch", "ckpt", "pipeline", "reset_after",
               "warmup", "warm_start", "distilled", "p", "pool"]


def check_one_recipe(runs: pd.DataFrame, what: str) -> None:
    # refuse to fit runs that mix training recipes
    keys = runs[RECIPE_KEYS].drop_duplicates()
    if len(keys) > 1:
        raise ValueError(f"{what}: runs mix recipes:\n{keys.to_string()}")


def select(df: pd.DataFrame, **kw) -> pd.DataFrame:
    # rows matching every keyword (floats with a tolerance)
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        col = df[k]
        if isinstance(v, (list, tuple, set)):
            m &= col.isin(list(v))
        elif isinstance(v, float):
            m &= np.isclose(col.astype(float), v, rtol=1e-6)
        else:
            m &= col.eq(v)
    return df[m].copy()


def load_pending(df: pd.DataFrame, extra_roots: list[Path] | None = None) -> pd.DataFrame:
    # Runs not yet scored (figures/data/pending_runs.csv). Validation p_L from the CSV, else
    # the lowest 'val_ler=' in the run.log matched by run_log_glob. Dropped once the
    # all-runs CSV has a scored run with the same d, width, data, LR, epochs, warmup.
    pend = pd.read_csv(DATA_DIR / "pending_runs.csv")
    roots = [REPO] + list(extra_roots or [])
    out = []
    for _, r in pend.iterrows():
        N = parse_shots(r.train_shots)
        warm = bool(r.warmup_epochs and r.warmup_epochs > 0)
        done = df[df.scored & df.d.eq(r.d) & df.width.eq(r.width) & np.isclose(df.N, N)
                  & np.isclose(df.lr, r.peak_lr) & df.epochs.eq(r.epochs) & df.warmup.eq(warm)]
        if len(done):
            continue  # scored now; the CSV row is used instead
        val, src = r.val_p_L, r.val_source
        if pd.isna(val) and isinstance(r.run_log_glob, str):
            for root in roots:
                logs = sorted(glob.glob(str(root / r.run_log_glob), recursive=True))
                if logs:
                    v = min_val_ler(Path(logs[-1]))
                    if v is not None:
                        val, src = v, f"lowest val_ler in {logs[-1]}"
                        break
        mwpm = MWPM_FALLBACK[int(r.d)]
        out.append(dict(
            csv_line=np.nan, status_raw=r.status, width=r.width, params=float(r.params),
            d=int(r.d), lr=float(r.peak_lr), data=r.train_shots, N=N, seed=0, p=0.004,
            batch=10000, epochs=int(r.epochs), schedule=r.schedule, ckpt="val p_L",
            pipeline="new", pool="grace1", warmup=warm, warm_start=False, distilled=False,
            reset_after=False,
            diagnostic=False, stability="killed" if r.status == "killed" else "stable",
            scored=False, val_only=pd.notna(val), p_L=val, mwpm=mwpm,
            n_shots=N_VAL, k=np.round(val * N_VAL) if pd.notna(val) else np.nan,
            ratio=val / mwpm if pd.notna(val) else np.nan, run_id=r.run_id,
            note=f"{r.note}; {src}" if pd.notna(val) else r.note,
            label=f"{r.run_id}: d{r.d} {r.width} {r.train_shots} cosine {fmt_lr(r.peak_lr)} {r.epochs}ep"))
    p = pd.DataFrame(out)
    if len(p):
        p["recipe"] = p.apply(recipe_string, axis=1)
    return p


def min_val_ler(log: Path) -> float | None:
    vals = [float(x) for x in re.findall(r"val_ler=([0-9.]+)", log.read_text(errors="ignore"))]
    return min(vals) if vals else None


def load_extra() -> pd.DataFrame:
    # train-slice, overfit-test and McNemar numbers not in the all-runs CSV
    e = pd.read_csv(DATA_DIR / "extra_scores.csv")
    e["k"] = np.round(e["p_L"] * e["shots"])
    return e


@dataclass
class Data:
    runs: pd.DataFrame
    pending: pd.DataFrame
    extra: pd.DataFrame
    source: str
    mwpm: dict = field(default_factory=dict)


def load_all(csv_path=None, extra_roots=None) -> Data:
    runs = load_runs(csv_path)
    mw = (runs[runs.scored & np.isclose(runs.p, 0.004)].groupby("d")["mwpm"]
          .agg(lambda s: s.mode().iloc[0]).to_dict())
    for d, v in mw.items():
        MWPM_FALLBACK[d] = v
    return Data(runs=runs, pending=load_pending(runs, extra_roots), extra=load_extra(),
                source=runs.attrs["source"], mwpm=dict(MWPM_FALLBACK))


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------
def wilson(k, n, z=1.959964):
    # Wilson score interval; array-safe, k may be 0
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    ph = k / n
    den = 1 + z ** 2 / n
    ctr = (ph + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / den
    return ctr - half, ctr + half


def power_law(x, E, B, beta):
    return E + B * np.power(x, -beta)


@dataclass
class Fit:
    name: str
    figure: str
    x_name: str
    E: float
    B: float
    beta: float
    E_ci: tuple
    B_ci: tuple
    beta_ci: tuple
    mwpm: float
    n_runs: int
    n_x: int
    boot_ok: int
    runs: str
    recipe: str
    x_scale: float

    def row(self) -> dict:
        r = lambda v: float(np.round(v, 4))
        return {
            "figure": self.figure, "series": self.name, "x": self.x_name,
            "model": f"p_L = E + B*(x/{fmt_shots(self.x_scale) if self.x_name == 'N' else fmt_params(self.x_scale)})^(-beta)",
            "E_over_MWPM": r(self.E / self.mwpm),
            "E_over_MWPM_lo": r(self.E_ci[0] / self.mwpm), "E_over_MWPM_hi": r(self.E_ci[1] / self.mwpm),
            "E": self.E, "E_lo": self.E_ci[0], "E_hi": self.E_ci[1],
            "beta": r(self.beta), "beta_lo": r(self.beta_ci[0]), "beta_hi": r(self.beta_ci[1]),
            "B": self.B, "B_lo": self.B_ci[0], "B_hi": self.B_ci[1],
            "mwpm": self.mwpm, "n_runs": self.n_runs, "n_distinct_x": self.n_x,
            "bootstrap_fits_converged": f"{self.boot_ok}/{N_BOOT}",
            "recipe": self.recipe, "runs": self.runs,
        }


def _fit_once(x, k, n):
    p = k / n
    # binomial SE, floored at one error so no run gets infinite weight
    sig = np.sqrt(np.maximum(p * (1 - p), 1 / n) / n)
    E0 = 0.9 * p.min()
    beta0 = 0.7
    B0 = max(p.max() - E0, 1e-6) * x.min() ** beta0
    popt, _ = curve_fit(power_law, x, p, p0=[E0, B0, beta0], sigma=sig, absolute_sigma=True,
                        bounds=([0, 0, 0.01], [1, np.inf, 5]), maxfev=20000)
    return popt


def fit_power(runs: pd.DataFrame, xcol: str, x_scale: float, name: str, figure: str,
              min_distinct=4) -> Fit | None:
    # Weighted least squares p_L = E + B (x/x_scale)^(-beta), seeds as separate points.
    # Bootstrap: resample seeds at each x, then redraw each run's errors from Binomial(n, k/n).
    runs = runs[runs.scored & runs.stability.eq("stable")]
    if runs[xcol].nunique() < min_distinct:
        return None
    check_one_recipe(runs, name)
    x = runs[xcol].to_numpy(float) / x_scale
    k = runs["k"].to_numpy(float)
    n = runs["n_shots"].to_numpy(float)
    E, B, beta = _fit_once(x, k, n)
    rng = np.random.default_rng(RNG_SEED)
    groups = [np.flatnonzero(np.isclose(x, xv)) for xv in np.unique(x)]
    boots = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        kb = rng.binomial(n[idx].astype(int), k[idx] / n[idx])
        try:
            boots.append(_fit_once(x[idx], kb.astype(float), n[idx]))
        except (RuntimeError, ValueError):
            pass
    b = np.array(boots)
    ci = lambda col: tuple(np.percentile(b[:, col], [2.5, 97.5]))
    return Fit(name=name, figure=figure, x_name=xcol, E=E, B=B, beta=beta, E_ci=ci(0),
               B_ci=ci(1), beta_ci=ci(2), mwpm=float(runs.mwpm.iloc[0]), n_runs=len(runs),
               n_x=runs[xcol].nunique(), boot_ok=len(b),
               runs="; ".join(f"line {int(r.csv_line)} ({r.label}, {r.ratio:.3f}x)"
                              for r in runs.sort_values([xcol, "seed"]).itertuples()),
               recipe=runs.recipe.iloc[0], x_scale=x_scale)


def mcnemar_stars(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


# --------------------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------------------
@dataclass
class FigResult:
    key: str
    fig: plt.Figure
    title: str
    caption: str
    runs_used: pd.DataFrame
    fits: list = field(default_factory=list)
    table: pd.DataFrame | None = None

    def fit_table(self) -> pd.DataFrame:
        if self.table is not None:
            return self.table
        if not self.fits:
            return pd.DataFrame({"note": ["no series in this figure has enough points with one recipe to fit"]})
        cols = ["series", "E_over_MWPM", "E_over_MWPM_lo", "E_over_MWPM_hi", "beta", "beta_lo",
                "beta_hi", "n_runs", "n_distinct_x", "bootstrap_fits_converged", "recipe", "runs"]
        return pd.DataFrame([f.row() for f in self.fits])[cols]


def common_caption(data: Data, extra_sources: bool = False) -> str:
    # shared last sentence of every caption
    src = data.source + (", figures/data/extra_scores.csv" if extra_sources else "")
    return f"Eval block: 1.8M shots per d, same MWPM for every run at a given d. Bars: Wilson 95%. Data: {src}."


def setup_ratio_axis(ax, lo, hi, label="p_L / MWPM"):
    # only lower the bottom, so already-plotted points stay visible
    y0 = ax.dataLim.y0
    if np.isfinite(y0) and 0 < y0 < lo / 0.9:
        lo = y0 * 0.9
    ax.set_yscale("log")
    cands = [0.3, 0.5, 0.7, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]
    ticks = [t for t in cands if lo <= t <= hi]
    if len(ticks) > 9:
        ticks = [t for t in ticks if t in (0.5, 0.8, 1, 1.25, 1.5, 2, 3, 5, 10, 20, 50, 100)]
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.set_ylabel(label)


def mwpm_line(ax, label="MWPM"):
    ax.axhline(1.0, color=MWPM_COLOR, ls="--", lw=1.0, zorder=1)
    ax.annotate(label, xy=(0.0, 1.0), xycoords=("axes fraction", "data"), xytext=(3, 3),
                textcoords="offset points", ha="left", va="bottom", fontsize=8.5)


def setup_log_x(ax, ticks, fmt, lo, hi, label):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))
    ax.set_xlabel(label)


SHOT_TICKS = [1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 2e8, 5e8]
PARAM_TICKS = [5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6]


def annotate_ratio(ax, x, y, text=None, dx=5, dy=4, ha="left", **kw):
    ax.annotate(text if text is not None else f"{y:.2f}x", (x, y), xytext=(dx, dy),
                textcoords="offset points", fontsize=8, ha=ha, va="bottom", color="#333333", **kw)


def plot_runs(ax, runs: pd.DataFrame, xcol: str, color, marker, label=None, ls="-",
              yfun=lambda r: r.ratio, connect=True, jitter=0.035, annotate=(), ann_kw=None,
              z=3, ms=6.5, labels=None):
    # One series: Wilson bar per run, seed mean as a larger marker, stable means joined.
    # 'x' = diverged / unstable / killed; hollow = validation only. Returns the line's (x, y).
    handles_xy = []
    if runs.empty:
        return handles_xy
    ok = runs[runs.stability.eq("stable") & ~runs.val_only]
    for xv, g in ok.groupby(xcol):
        y = np.array([yfun(r) for r in g.itertuples()])
        lo, hi = wilson(g.k.to_numpy(float), g.n_shots.to_numpy(float))
        scale = (y / g.p_L.to_numpy()) if len(g) else 1
        lo, hi = lo * scale, hi * scale
        if len(g) == 1:
            ax.errorbar([xv], y, yerr=[y - lo, hi - y], fmt=marker, color=color, ms=ms,
                        mec="white", mew=0.6, elinewidth=1.1, zorder=z)
        else:
            offs = xv * np.exp(np.linspace(-jitter, jitter, len(g)))
            ax.errorbar(offs, y, yerr=[y - lo, hi - y], fmt=marker, color=color, ms=3.3,
                        alpha=0.75, elinewidth=0.8, mew=0, zorder=z)
            ax.plot([xv], [y.mean()], marker=marker, color=color, ms=ms + 0.5, mec="white",
                     mew=0.6, ls="none", zorder=z + 1)
        handles_xy.append((xv, y.mean()))
        if any(np.isclose(xv, a) for a in annotate):
            if labels is not None:
                labels.append((xv, y.mean(), f"{y.mean():.2f}x"))
            else:
                annotate_ratio(ax, xv, y.mean(), **(ann_kw or {}))
    if connect and len(handles_xy) > 1:
        xs, ys = zip(*sorted(handles_xy))
        ax.plot(xs, ys, ls=ls, color=color, lw=1.4, zorder=z - 1)
    bad = runs[~runs.stability.eq("stable") & (runs.scored | runs.val_only)]
    for r in bad.itertuples():
        y = yfun(r)
        ax.plot([getattr(r, xcol)], [y], marker="x", color=color, ms=7, mew=1.8, ls="none", zorder=z + 1)
        tag = {"diverged": "diverged", "unstable": "unstable", "killed": "killed"}[r.stability]
        if r.val_only:
            tag += ", val-only"
        if labels is not None:
            labels.append((getattr(r, xcol), y, tag))
        else:
            ax.annotate(tag, (getattr(r, xcol), y), xytext=(6, -3), textcoords="offset points",
                        fontsize=7.5, color="#333333", va="top")
    val = runs[runs.val_only & runs.stability.eq("stable")]
    for r in val.itertuples():
        y = yfun(r)
        lo, hi = wilson(r.k, r.n_shots)
        s = y / r.p_L
        ax.errorbar([getattr(r, xcol)], [y], yerr=[[y - lo * s], [hi * s - y]], fmt=marker,
                    mfc="white", mec=color, color=color, ms=ms, mew=1.3, elinewidth=0.9, zorder=z)
        if labels is not None:
            labels.append((getattr(r, xcol), y, f"{r.run_id}: pending, val-only"))
        else:
            ax.annotate(f"{r.run_id}: pending, val-only", (getattr(r, xcol), y), xytext=(6, 2),
                        textcoords="offset points", fontsize=7.5, color="#333333")
    return handles_xy


def place_labels(ax, items, fontsize=7.5, avoid_points=()):
    # Label points without overlapping each other, markers or the axes edge.
    # items: (x, y, text) in data coordinates. Call after limits and layout are final.
    fig = ax.figure
    r = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(r)
    to_px = lambda x, y: ax.transData.transform((x, y))
    mk = [Bbox.from_extents(px - 4, py - 4, px + 4, py + 4)
          for px, py in (to_px(x, y) for x, y, *_ in list(items) + list(avoid_points))]
    placed = []
    cands = [(6, 0, "left", "center"), (-6, 0, "right", "center"), (0, 6, "center", "bottom"),
             (0, -6, "center", "top"), (6, 7, "left", "bottom"), (6, -7, "left", "top"),
             (-6, 7, "right", "bottom"), (-6, -7, "right", "top"), (10, 14, "left", "bottom"),
             (10, -14, "left", "top"), (-10, 14, "right", "bottom"), (-10, -14, "right", "top")]
    for x, y, t in items:
        for i, (dx, dy, ha, va) in enumerate(cands):
            a = ax.annotate(t, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=fontsize, color="#333333", zorder=7)
            bb = a.get_window_extent(r)
            inside = (bb.x0 >= axbb.x0 and bb.x1 <= axbb.x1 and bb.y0 >= axbb.y0 and bb.y1 <= axbb.y1)
            if inside and not any(bb.overlaps(o) for o in placed + mk):
                placed.append(bb)
                break
            if i < len(cands) - 1:
                a.remove()
        else:
            placed.append(bb)


def legend_handle(color, marker, ls="-", label="", hollow=False):
    return Line2D([], [], color=color, marker=marker, ls=ls, ms=6, label=label,
                  mfc="white" if hollow else color, mec=color if hollow else "white")


def finish(fig, title: str, caption: str, below_axes_in=0.6):
    # Title on top, wrapped caption below. Returns the bottom fraction for subplots_adjust
    # (below_axes_in ~0.1 when the lowest axes is a table).
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    lines = textwrap.fill(caption, width=int(fig.get_figwidth() * 14.5))
    fig.text(0.01, 0.01, lines, ha="left", va="bottom", fontsize=8.5, color="#333333",
             linespacing=1.3)
    n = lines.count("\n") + 1
    caption_in = n * 8.5 * 1.3 / 72 + 0.12
    return (caption_in + below_axes_in) / fig.get_figheight()


def summary_md(r: "FigResult") -> str:
    # notebook text for one figure: takeaway, then every run used, built from the same runs
    lines = [r.title, "", "<details><summary>Runs</summary>", ""]
    ru = r.runs_used
    if ru is not None and len(ru) and "series" not in ru.columns:
        # figure 4: one line per (run, checkpoint, block) from figures/data/extra_scores.csv
        for x in ru.itertuples():
            v = f"{x.ratio:.2f}x" if x.ratio == x.ratio else f"p_L {x.p_L:.4f}"
            lines.append(f"- {x.run_id}, {x.checkpoint} ckpt, {x.block}: {v}")
    elif ru is not None and len(ru):
        for series, g in ru.groupby("series", sort=False):
            items = []
            for x in g.itertuples():
                tag = f"{x.width} {x.train_shots} s{x.seed}"
                if x.block == "val (pending)":
                    items.append(f"{tag} pending (val {x.ratio:.2f}x)" if x.ratio == x.ratio else f"{tag} pending")
                else:
                    items.append(f"{tag} {x.ratio:.2f}x" + (" (x)" if x.stability != "stable" else "")
                                 if x.ratio == x.ratio and x.ratio is not None else f"{tag} not scored")
            lines.append(f"- {series}: " + ", ".join(items))
    lines += ["", "</details>"]
    return "\n".join(lines)


def runs_used_table(key, groups: list[tuple[str, pd.DataFrame, str]]) -> pd.DataFrame:
    # flatten (series, runs, role) triples for runs_used.csv
    rows = []
    for series, g, role in groups:
        for r in g.itertuples():
            rows.append({
                "figure": key, "series": series, "role": role,
                "csv_line": None if pd.isna(getattr(r, "csv_line", np.nan)) else int(r.csv_line),
                "run": r.label, "d": r.d, "width": r.width, "params": int(r.params),
                "train_shots": r.data, "seed": r.seed, "p_L": r.p_L, "ratio": round(r.ratio, 4) if pd.notna(r.ratio) else None,
                "block": "val (pending)" if r.val_only else "eval",
                "stability": r.stability, "recipe": r.recipe})
    return pd.DataFrame(rows)


def draw_table(ax, df: pd.DataFrame, col_widths=None, fontsize=7.5):
    ax.axis("off")
    if df.empty:
        return
    t = ax.table(cellText=df.values, colLabels=list(df.columns), loc="upper left",
                 cellLoc="left", colLoc="left", colWidths=col_widths, edges="horizontal")
    t.auto_set_font_size(False)
    t.set_fontsize(fontsize)
    t.scale(1, 1.25)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor("#BBBBBB")
        c.set_linewidth(0.5)
        if i == 0:
            c.set_text_props(fontweight="bold")


def fmt_ci(v, lo, hi, nd=2):
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


# --------------------------------------------------------------------------------------
# Figure 1: data scaling
# --------------------------------------------------------------------------------------
def series_data_scaling(data: Data) -> list[dict]:
    # fixed-recipe series of Figure 1 (data size varies)
    R, P = data.runs, data.pending
    S = []
    S.append(dict(name="d7 u280 cosine 3e-3, 80 ep", width="u280", d=7, ls="-", fit=True,
                  runs=select(R, d=7, width="u280", schedule="cosine", lr=0.003, epochs=80,
                              p=0.004, warmup=False, pool="grace1", diagnostic=False)))
    S.append(dict(name="d7 u200 cosine 3e-3, 80 ep", width="u200", d=7, ls="-", fit=True,
                  runs=select(R, d=7, width="u200", schedule="cosine", lr=0.003, epochs=80,
                              p=0.004, warmup=False, diagnostic=False)))
    S.append(dict(name="d7 u140 constant 3e-3 (older pipeline, val-loss ckpt)", width="u140", d=7,
                  ls="--", fit=True,
                  runs=select(R, d=7, width="u140", schedule="constant", lr=0.003, p=0.004,
                              pipeline="old", ckpt="val loss", reset_after=False, scored=True)))
    S.append(dict(name="d7 u140 constant 3e-3 (new pipeline, val-p_L ckpt)", width="u140", d=7,
                  ls="none", fit=False,
                  runs=select(R, d=7, width="u140", schedule="constant", lr=0.003, p=0.004,
                              pipeline="new", ckpt="val p_L", reset_after=False, scored=True)))
    S.append(dict(name="d5 u140 constant 3e-3 (older pipeline, val-loss ckpt)", width="u140", d=5,
                  ls="--", fit=True,
                  runs=select(R, d=5, width="u140", schedule="constant", lr=0.003, p=0.004,
                              pipeline="old", ckpt="val loss", reset_after=False, scored=True)))
    S.append(dict(name="d9 u280 cosine 3e-3, 80 ep", width="u280", d=9, ls="-", fit=True,
                  runs=select(R, d=9, width="u280", schedule="cosine", lr=0.003, epochs=80,
                              warmup=False)))
    d9w = select(R, d=9, width="u280", schedule="cosine", lr=0.001, epochs=160, warmup=True,
                 warm_start=False, diagnostic=False)
    d9w = d9w[d9w.scored]
    if len(P):
        d9w = pd.concat([d9w, select(P, d=9, width="u280", lr=0.001, epochs=160)])
    S.append(dict(name="d9 u280 cosine 1e-3 + warmup, 160 ep", width="u280", d=9, ls=":",
                  fit=True, runs=d9w))
    return S


def fig1_data_scaling(data: Data) -> FigResult:
    S = series_data_scaling(data)
    fig = plt.figure(figsize=(7.2, 8.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.3, 0.75, 0.5], hspace=0.3)
    ax = fig.add_subplot(gs[0])
    lax = fig.add_subplot(gs[1])
    tax = fig.add_subplot(gs[2])
    lax.axis("off")
    fits, handles, groups, labels = [], [], [], []
    x_hi = 3.2e8
    for s in S:
        runs = s["runs"]
        if runs.empty:
            continue
        col, mk = WIDTH_COLOR[s["width"]], D_MARKER[s["d"]]
        role = "plotted"
        if s["fit"]:
            f = fit_power(runs, "N", 1e6, s["name"], "fig1")
            if f:
                fits.append(f)
                role = "fit"
                # fit curve over the fitted range, extended to the right edge
                xx = np.logspace(np.log10(runs.N.min() / 1.15), np.log10(x_hi), 150)
                ax.plot(xx, power_law(xx / 1e6, f.E, f.B, f.beta) / f.mwpm, color=col, lw=0.9,
                        alpha=0.8, ls=(0, (1, 1.5)), zorder=2)
                # asymptote E with its 95% CI, as a bracket at the right edge
                xb = x_hi * 0.93
                lo, hi = f.E_ci[0] / f.mwpm, f.E_ci[1] / f.mwpm
                ax.plot([xb, xb], [max(lo, 0.81), hi], color=col, lw=5, alpha=0.35,
                        solid_capstyle="butt", zorder=2)
                ax.plot([xb], [f.E / f.mwpm], marker="_", color=col, ms=10, mew=2, zorder=3)
        stable = runs[runs.scored & runs.stability.eq("stable")]
        ann = list(stable.N.unique()) if s["d"] in (7, 9) and s["width"] == "u280" else (
            [stable.N.max()] if len(stable) else [])
        plot_runs(ax, runs, "N", col, mk, ls=s["ls"] if s["ls"] != "none" else "-",
                  connect=s["ls"] != "none", annotate=ann, labels=labels)
        groups.append((s["name"], stable, role))
        groups.append((s["name"], runs[~(runs.scored & runs.stability.eq("stable"))], "plotted, excluded from fit"))
        handles.append(legend_handle(col, mk, ls=s["ls"] if s["ls"] != "none" else "", label=s["name"]))
    mwpm_line(ax)
    setup_log_x(ax, SHOT_TICKS, fmt_shots, 1.5e6, x_hi, "training shots N")
    ys = pd.concat([s["runs"] for s in S])["ratio"].dropna()
    setup_ratio_axis(ax, 0.8, max(12, ys.max() * 1.2))
    handles += [Line2D([], [], color="#555555", lw=0.9, ls=(0, (1, 1.5)), label="fit E + B N^-beta"),
                Line2D([], [], color="#555555", lw=5, alpha=0.35, label="asymptote E, 95% CI (right edge)"),
                Line2D([], [], color="#555555", marker="x", ls="", label="diverged / unstable (not fitted)")]
    lax.legend(handles=handles, loc="center", ncol=2, fontsize=7.5)

    rows = [[f.name, fmt_ci(f.E / f.mwpm, f.E_ci[0] / f.mwpm, f.E_ci[1] / f.mwpm),
             fmt_ci(f.beta, *f.beta_ci), f"{f.n_runs} / {f.n_x}"] for f in fits]
    draw_table(tax, pd.DataFrame(rows, columns=["series (fitted)", "E (x MWPM) [95% CI]",
                                                "beta [95% CI]", "runs / distinct N"]),
               col_widths=[0.46, 0.2, 0.2, 0.14])

    head = S[0]["runs"]
    head = head[head.scored & head.stability.eq("stable")].groupby("N").ratio.mean()
    f0 = next((f for f in fits if f.name == S[0]["name"]), None)
    title = (f"d7 u280: {head.iloc[0]:.2f}x MWPM at {fmt_shots(head.index[0])}, "
             f"{head.loc[1e8]:.2f}x at 100M, {head.iloc[-1]:.2f}x at {fmt_shots(head.index[-1])}"
             + (f"; fitted floor {f0.E / f0.mwpm:.2f}x" if f0 else ""))
    pend = data.pending
    missing = pend[~pend.val_only & pend.d.eq(9)].run_id.tolist() if len(pend) else []
    missing = [m for m in missing if m == "9-5"]
    cap = ("Each line = one recipe; only N changes. Solid: cosine 3e-3, 80 ep. Dashed: constant 3e-3, older "
           "pipeline. Dotted: d9 1e-3 + warmup, 160 ep. Fit E + B N^-beta where >= 4 N; bar at right = E, 95% CI. "
           + (f"Pending: {', '.join(missing)}. " if missing else "") + common_caption(data))
    b = finish(fig, title, cap, below_axes_in=0.35)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=b)
    place_labels(ax, labels)
    return FigResult("fig1_data_scaling", fig, title, cap, runs_used_table("fig1", groups), fits)


# --------------------------------------------------------------------------------------
# Figure 2: model-size scaling
# --------------------------------------------------------------------------------------
def _one_layer(runs: pd.DataFrame, single: bool) -> pd.DataFrame:
    # one GRU layer (single=True) or stacked layers ('2xu280' etc.)
    if runs.empty:
        return runs
    stacked = runs.width.astype(str).str.contains("x")
    return runs[~stacked] if single else runs[stacked]


def fig2_model_size(data: Data) -> FigResult:
    R, P = data.runs, data.pending
    S = [
        dict(name="d7 cosine 3e-3, 80 ep, 100M", d=7, ls="-", runs=select(
            R, d=7, schedule="cosine", lr=0.003, epochs=80, N=1e8, p=0.004, warmup=False, scored=True)),
        dict(name="d7 cosine 3e-3, 80 ep, 200M", d=7, ls="-.", runs=select(
            R, d=7, schedule="cosine", lr=0.003, epochs=80, N=2e8, p=0.004, warmup=False, scored=True)),
        # single-layer and stacked GRUs are separate series, never fitted together
        dict(name="d7 cosine 1e-3 + warmup, 80 ep, 200M, one GRU layer (C1, arm a)", d=7, ls=":",
             runs=_one_layer(pd.concat([
                 select(R, d=7, schedule="cosine", lr=0.001, epochs=80, N=2e8, warmup=True,
                        scored=True, distilled=False),
                 select(P, d=7, lr=0.001, N=2e8) if len(P) else pd.DataFrame()]), True)),
        dict(name="d7 cosine 1e-3 + warmup, 80 ep, 200M, two stacked layers (S1, S2, arm b)", d=7,
             ls=(0, (6, 2, 1, 2)),
             runs=_one_layer(pd.concat([
                 select(R, d=7, schedule="cosine", lr=0.001, epochs=80, N=2e8, warmup=True,
                        scored=True, distilled=False),
                 select(P, d=7, lr=0.001, N=2e8) if len(P) else pd.DataFrame()]), False)),
        dict(name="d9 cosine 3e-3, 80 ep, 100M", d=9, ls="-", runs=select(
            R, d=9, schedule="cosine", lr=0.003, epochs=80, N=1e8, warmup=False, scored=True)),
        dict(name="d7 constant 3e-3, 200 ep, 20M (val-p_L ckpt)", d=7, ls="--", runs=select(
            R, d=7, schedule="constant", lr=0.003, N=2e7, p=0.004, pipeline="new", ckpt="val p_L",
            reset_after=False, scored=True)),
        dict(name="d7 u140 constant 20M, older pipeline", d=7, ls="none", runs=select(
            R, d=7, width="u140", schedule="constant", lr=0.003, N=2e7, pipeline="old", reset_after=False,
            scored=True)),
    ]
    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 0.75], hspace=0.2)
    ax = fig.add_subplot(gs[0])
    lax = fig.add_subplot(gs[1])
    lax.axis("off")
    budget = select(R, width="u280").params.max()
    ax.axvspan(budget * 1.04, 3e6, color="#F2F2F2", lw=0, zorder=0)
    ax.text(budget * 1.12, 0.97, "diagnostic only\n(exceeds FPGA budget)", transform=ax.get_xaxis_transform(),
            va="top", fontsize=8, color="#555555")
    fits, groups, handles, labels = [], [], [], []
    offsets = {}
    for i, s in enumerate(S):
        runs = s["runs"]
        if runs.empty:
            continue
        runs = runs.copy()
        # small x offset per series so markers at the same parameter count do not overlap
        off = np.exp((i - 2.5) * 0.025)
        runs["x"] = runs["params"] * off
        f = fit_power(runs.assign(x=runs["params"]), "x", 1e5, s["name"], "fig2") if s["ls"] != "none" else None
        if f:
            fits.append(f)
            xx = np.logspace(np.log10(runs.params.min() * 0.8), np.log10(runs.params.max() * 1.2), 100)
            ax.plot(xx, power_law(xx / 1e5, f.E, f.B, f.beta) / f.mwpm, color="#555555", lw=0.9)
        # points coloured by width, joined by a neutral line
        stable = runs[(runs.scored | runs.val_only) & runs.stability.eq("stable")]
        means = stable.groupby("x").ratio.mean().sort_index()
        if s["ls"] != "none" and len(means) > 1:
            ax.plot(means.index, means.values, ls=s["ls"], color="#888888", lw=1.1, zorder=2)
        for w, g in runs.groupby("width"):
            plot_runs(ax, g, "x", WIDTH_COLOR[w], D_MARKER[s["d"]], connect=False,
                      annotate=list(g.x), labels=labels)
        groups.append((s["name"], runs, "fit" if f else "plotted"))
        handles.append(Line2D([], [], color="#888888", ls=s["ls"] if s["ls"] != "none" else "",
                              marker=D_MARKER[s["d"]], mfc="#888888", mec="white", label=s["name"]))
    mwpm_line(ax)
    xs = pd.concat([s["runs"] for s in S]).params
    setup_log_x(ax, PARAM_TICKS, lambda v: fmt_params(v) + (" params" if np.isclose(v, 1e5) else ""),
                xs.min() * 0.7, max(xs.max() * 1.6, 1.3e6), "parameters (actual count)")
    ys = pd.concat([s["runs"] for s in S]).ratio.dropna()
    setup_ratio_axis(ax, 0.85, ys.max() * 1.25)
    whandles = [legend_handle(c, "o", ls="", label=w) for w, c in WIDTH_COLOR.items()
                if w in set(pd.concat([s["runs"] for s in S]).width)]
    leg1 = lax.legend(handles=handles, loc="center left", fontsize=7.3, title="series (line style; marker = d)",
                      title_fontsize=7.5)
    lax.add_artist(leg1)
    lax.legend(handles=whandles, loc="center right", fontsize=7.5, title="width (colour)", title_fontsize=7.5,
               ncol=2)
    pend = P[~P.val_only & P.d.eq(7)].run_id.tolist() if len(P) else []
    best100 = S[0]["runs"].sort_values("params")
    title = ("At fixed data and recipe, d7 error falls with width: "
             + ", ".join(f"{r.width} {r.ratio:.2f}x" for r in best100.itertuples())
             + " at 100M")
    cap = ("Each line = one recipe at one data size. Shaded = above the u280 FPGA budget. "
           + (f"Pending: {', '.join(pend)}. " if pend else "") + common_caption(data))
    b = finish(fig, title, cap, below_axes_in=0.1)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.94, bottom=b)
    place_labels(ax, labels)
    return FigResult("fig2_model_size", fig, title, cap, runs_used_table("fig2", groups), fits)


# --------------------------------------------------------------------------------------
# Figure 3: error suppression vs distance
# --------------------------------------------------------------------------------------
def _lambda_boot(series_runs: dict, rng) -> np.ndarray:
    # bootstrap p_L per d: resample seeds, redraw errors, average -> (N_BOOT, n_d)
    ds = sorted(series_runs)
    out = np.empty((N_BOOT, len(ds)))
    for j, d in enumerate(ds):
        g = series_runs[d]
        k, n = g.k.to_numpy(float), g.n_shots.to_numpy(float)
        for i in range(N_BOOT):
            idx = rng.integers(0, len(g), len(g))
            out[i, j] = (rng.binomial(n[idx].astype(int), k[idx] / n[idx]) / n[idx]).mean()
    return out


def fig3_distance(data: Data) -> FigResult:
    R = data.runs
    base = R[R.scored & R.stability.eq("stable") & np.isclose(R.p, 0.004)]
    # within the FPGA budget: at most the single u280 layer's parameters at that d
    cap = R[R.width.eq("u280")].groupby("d").params.max()
    base = base[base.params <= base.d.map(cap).fillna(np.inf)]
    lines = [
        dict(name="u140 constant 3e-3, 10M (older pipeline)", width="u140", ls="--",
             runs=select(base, width="u140", schedule="constant", lr=0.003, N=1e7, pipeline="old")),
        dict(name="u280 constant 3e-3, 10M (older pipeline)", width="u280", ls="--",
             runs=select(base, width="u280", schedule="constant", lr=0.003, N=1e7, pipeline="old")),
        dict(name="u280 cosine 3e-3, 80 ep, 100M", width="u280", ls="-",
             runs=select(base, width="u280", schedule="cosine", lr=0.003, epochs=80, N=1e8, warmup=False)),
    ]
    for l in lines:
        for d, g in l["runs"].groupby("d"):
            check_one_recipe(g.assign(d=0), l["name"])
        check_one_recipe(l["runs"], l["name"])
    best = base.loc[base.groupby("d").p_L.idxmin()].sort_values("d")
    best_labels = []
    rng = np.random.default_rng(RNG_SEED)

    fig = plt.figure(figsize=(7.2, 6.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 0.8], hspace=0.2)
    ax = fig.add_subplot(gs[0])
    tax = fig.add_subplot(gs[1])
    ds = sorted(data.mwpm)
    mw = np.array([data.mwpm[d] for d in ds])
    mlo, mhi = wilson(np.round(mw * N_EVAL), N_EVAL)
    ax.errorbar(ds, mw, yerr=[mw - mlo, mhi - mw], color=MWPM_COLOR, lw=1.8, marker="D", ms=5,
                label="MWPM", zorder=5)
    # best GRU per distance (envelope, any recipe)
    blo, bhi = wilson(best.k, best.n_shots)
    ax.errorbar(best.d, best.p_L, yerr=[best.p_L - blo, bhi - best.p_L], color=NEUTRAL, lw=2.6,
                alpha=0.55, zorder=3, label="best GRU per d (any recipe)")
    for r in best.itertuples():
        ax.plot(r.d, r.p_L, marker=D_MARKER[r.d], color=WIDTH_COLOR[r.width], ms=9, mec="black", mew=0.8, zorder=6)
        best_labels.append((r.d, r.p_L, f"best: {r.width} {r.data} {r.ratio:.2f}x"))
    groups = [("best GRU per d", best, "plotted")]
    for l in lines:
        g = l["runs"]
        col = WIDTH_COLOR[l["width"]]
        for d, gd in g.groupby("d"):
            plot_runs(ax, gd.assign(dx=d + (0.08 if l["ls"] == "-" else -0.08)), "dx", col,
                      D_MARKER[d], yfun=lambda r: r.p_L, connect=False, jitter=0.004, ms=5.5)
        m = g.groupby("d").p_L.mean()
        ax.plot(m.index + (0.08 if l["ls"] == "-" else -0.08), m.values, ls=l["ls"], color=col, lw=1.3,
                label=l["name"], zorder=2)
        groups.append((l["name"], g, "plotted (Lambda)"))
    ax.set_yscale("log")
    ax.set_xticks(ds)
    ax.set_xticklabels([f"d = {d}" for d in ds])
    ax.set_xlim(4.5, 9.9)
    ax.set_ylabel("logical error rate p_L (per shot, r = d rounds)")
    ax.yaxis.set_major_locator(FixedLocator([0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(loc="upper left", fontsize=7.5)

    # Lambda table with bootstrap CIs
    rows, lam_rows = [], []

    def add_lambda(name, per_d: dict):
        dd = sorted(per_d)
        boot = _lambda_boot(per_d, rng)
        pts = np.array([per_d[d].p_L.mean() for d in dd])
        for j in range(len(dd) - 1):
            if dd[j + 1] - dd[j] != 2:
                continue
            lam = pts[j] / pts[j + 1]
            bl = boot[:, j] / boot[:, j + 1]
            lo, hi = np.percentile(bl, [2.5, 97.5])
            lam_rows.append({"series": name, "from_d": dd[j], "to_d": dd[j + 1], "Lambda": lam,
                             "Lambda_lo": lo, "Lambda_hi": hi,
                             "runs": "; ".join(f"line {int(r.csv_line)} ({r.label})"
                                               for d in (dd[j], dd[j + 1]) for r in per_d[d].itertuples()
                                               if pd.notna(r.csv_line))})
    mwpm_runs = {d: pd.DataFrame({"p_L": [data.mwpm[d]], "k": [np.round(data.mwpm[d] * N_EVAL)],
                                  "n_shots": [N_EVAL], "csv_line": [np.nan], "label": ["MWPM"]}) for d in ds}
    add_lambda("MWPM", mwpm_runs)
    add_lambda("best GRU per d", {r.d: best[best.d.eq(r.d)] for r in best.itertuples()})
    for l in lines:
        add_lambda(l["name"], {d: g for d, g in l["runs"].groupby("d")})
    lam = pd.DataFrame(lam_rows)
    for name, g in lam.groupby("series", sort=False):
        cells = {f"{r.from_d}->{r.to_d}": fmt_ci(r.Lambda, r.Lambda_lo, r.Lambda_hi) for r in g.itertuples()}
        rows.append([name, cells.get("5->7", "-"), cells.get("7->9", "-")])
    draw_table(tax, pd.DataFrame(rows, columns=["series", "Lambda 5->7 [95% CI]", "Lambda 7->9 [95% CI]"]),
               col_widths=[0.5, 0.25, 0.25])

    lm = lam.set_index(["series", "from_d"])
    g79 = lm.loc[("best GRU per d", 7), "Lambda"]
    m79 = lm.loc[("MWPM", 7), "Lambda"]
    g57, m57 = lm.loc[("best GRU per d", 5), "Lambda"], lm.loc[("MWPM", 5), "Lambda"]
    title = (f"GRU \u039b < MWPM \u039b (5\u21927: {g57:.2f} vs {m57:.2f}; 7\u21929: {g79:.2f} vs {m79:.2f})"
             + ("; GRU \u039b < 1 from d7 to d9" if g79 < 1 else ""))
    best_desc = "; ".join(f"d{r.d}: {r.label}" + (" + warmup" if r.warmup else "") for r in best.itertuples())
    cap = ("Lambda = p_L(d) / p_L(d+2), bootstrap 95% CI. Grey = best GRU at each d (any recipe within the u280 parameter budget); "
           "coloured lines use one recipe at every d. " + common_caption(data))
    b = finish(fig, title, cap, below_axes_in=0.35)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.94, bottom=b)
    # label offsets for the best-GRU points
    for x, y, t in best_labels:
        dy = 12 if x == 7 else -12
        ax.annotate(t, (x, y), xytext=(0, dy), textcoords="offset points", ha="center",
                    va="bottom" if dy > 0 else "top", fontsize=7.5, color="#333333",
                    bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85), zorder=7)
    res = FigResult("fig3_distance", fig, title, cap, runs_used_table("fig3", groups))
    res.table = lam.assign(**{c: lam[c].round(3) for c in ["Lambda", "Lambda_lo", "Lambda_hi"]})
    return res


# --------------------------------------------------------------------------------------
# Figure 4: training slice vs unseen shots
# --------------------------------------------------------------------------------------
def _dumbbell(ax, E, rows, xticks, xlim):
    # One row per (label, run, checkpoint, block): grey = training slice, coloured = other
    # block, with ratio and McNemar stars vs MWPM on the same shots.
    n = len(rows)
    for i, (label, run, ck_tr, blk, ck_ot) in enumerate(rows):
        y = n - 1 - i
        tr = E[(E.run_id == run) & (E.block == "train") & (E.checkpoint == ck_tr)].iloc[0]
        ot = E[(E.run_id == run) & (E.block == blk) & (E.checkpoint == ck_ot)]
        mk, col = D_MARKER[int(tr.d)], WIDTH_COLOR[tr.width]
        xs = []
        for row, c, side in ((tr, NEUTRAL, "left"), (ot.iloc[0] if len(ot) else None, col, "right")):
            if row is None or pd.isna(row.ratio):
                continue
            lo, hi = wilson(row.k, row.shots)
            lo, hi = lo / row.mwpm_p_L, hi / row.mwpm_p_L
            if row.ratio == 0:
                # zero errors: Wilson upper bound with a leftward arrow
                ax.annotate("", xy=(hi / 4, y), xytext=(hi, y), arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
                ax.plot(hi, y, marker="|", color=c, ms=10, mew=1.8)
                txt, xv, side = f"0 errors in 2M shots (< {hi:.1g}x)", hi, "right"
            else:
                ax.errorbar(row.ratio, y, xerr=[[row.ratio - lo], [hi - row.ratio]], fmt=mk, color=c, ms=7,
                            mec="white", mew=0.6, zorder=4)
                if pd.notna(row.gru_only_b):
                    st = mcnemar_stars(row.p_exact)
                    win = "" if st == "n.s." else (" GRU" if row.gru_only_b > row.mwpm_only_c else " MWPM")
                    txt = f"{row.ratio:.2f}x {st}{win}"
                else:
                    txt = f"{row.ratio:.2f}x  (McNemar n/a)"
                xv = row.ratio
            xs.append(xv)
            ax.annotate(txt, (xv, y), xytext=(-8 if side == "left" else 8, 0), textcoords="offset points",
                        ha="right" if side == "left" else "left", va="center", fontsize=7.3, color="#333333",
                        bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85), zorder=6)
        if len(xs) == 2:
            ax.plot(xs, [y, y], color="#C8C8C8", lw=2.4, zorder=1, solid_capstyle="round")
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8)
    ax.set_ylim(-0.6, n - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.axvline(1, color="black", ls="--", lw=1)
    ax.annotate("MWPM", xy=(1, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -2),
                textcoords="offset points", fontsize=8, va="top")
    ax.set_xlabel("p_L / MWPM on the same shots")


def fig4_generalization(data: Data) -> FigResult:
    E = data.extra
    fig = plt.figure(figsize=(7.2, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 4], hspace=0.62)
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    # runs scored on both their training shots and the eval block (order of extra_scores.csv)
    both = E[(E.checkpoint == "best") & ~E.run_id.str.startswith("overfit")]
    names = [n for n in dict.fromkeys(both.run_id)
             if {"train", "eval"} <= set(both[both.run_id == n].block)]
    lo_ratio = both[both.run_id.isin(names)].ratio.min()
    _dumbbell(a1, E, [(n, n, "best", "eval", "best") for n in names],
              [0.5, 0.7, 0.8, 1, 1.25, 1.5, 2, 3], (min(0.42, lo_ratio * 0.6), 5.0))
    a1.set_title("a  Best checkpoint: own training shots vs unseen eval block", loc="left", fontsize=9.5)
    d7o, d9o = "overfit d7 u280 2M 300ep", "overfit d9 u280 2M 300ep"
    _dumbbell(a2, E, [("d7 best (val-chosen)", d7o, "best", "eval", "best"),
                      ("d7 last epoch", d7o, "last", "none", "last"),
                      ("d9 best (val-chosen)", d9o, "best", "eval", "best"),
                      ("d9 last epoch", d9o, "last", "none", "last")],
              [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000], (5e-5, 4000))
    a2.set_title("b  Memorization: u280 trained 300 epochs on 2M shots (diagnostic)", loc="left", fontsize=9.5)
    val_last = E[(E.block == "val") & (E.checkpoint == "last")]
    a2.text(0.0, -0.3, "Last-epoch weights on unseen validation shots: "
            + "; ".join(f"{r.run_id.split()[1]} p_L {r.p_L:.3f}" for r in val_last.itertuples())
            + " (vs. 0.004-0.005 for the 100M runs)", transform=a2.transAxes, fontsize=7.5, color="#333333")
    handles = [Line2D([], [], color=NEUTRAL, marker="o", ls="", label="own training shots [0, 2M)"),
               Line2D([], [], color=WIDTH_COLOR["u280"], marker="o", ls="", label="eval block [15.2M, 17M) (unseen)")]
    a1.legend(handles=handles, loc="lower right", fontsize=7.5, bbox_to_anchor=(1.0, 1.1), ncol=2)
    tr = E[(E.block == "train") & (E.checkpoint == "best")]
    ev = E[E.block == "eval"]
    gaps = {n: ev[ev.run_id == n].p_L.iloc[0] / tr[tr.run_id == n].p_L.iloc[0] - 1 for n in names}
    title = (f"Unseen p_L is {min(gaps.values()) * 100:.0f}-{max(gaps.values()) * 100:.0f}% above "
             f"own-training-shot p_L; 300 epochs on 2M shots memorizes")
    cap = ("(a) Best checkpoint on its own training shots vs the eval block; * = above the u280 FPGA "
           "parameter budget (diagnostic); stars = McNemar vs MWPM "
           "(*** p<1e-3). (b) Diagnostic: trained 300 ep on 2M shots only. " + common_caption(data, extra_sources=True))
    b = finish(fig, title, cap, below_axes_in=0.9)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.89, bottom=b)
    return FigResult("fig4_generalization", fig, title, cap,
                     E[["run_id", "checkpoint", "block", "shots", "p_L", "mwpm_p_L", "ratio", "gru_only_b",
                        "mwpm_only_c", "p_exact", "source"]].assign(figure="fig4"))


# --------------------------------------------------------------------------------------
# Figure 5: comparison panels
# --------------------------------------------------------------------------------------
def fig5_comparisons(data: Data) -> FigResult:
    R, P = data.runs, data.pending
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 7.0), gridspec_kw=dict(hspace=0.55, wspace=0.38))
    groups = []
    col = WIDTH_COLOR["u280"]

    # (a) peak LR
    ax = axs[0, 0]
    a7 = select(R, d=7, width="u280", schedule="cosine", epochs=80, N=5e7, p=0.004, warmup=False)
    a9 = select(R, d=9, width="u280", schedule="cosine", epochs=160, N=1e8, warmup=True, warm_start=False)
    a9 = a9[a9.scored | a9.stability.ne("stable")]
    a9 = a9[a9.scored]
    if len(P):
        a9 = pd.concat([a9, select(P, d=9, width="u280", N=1e8, epochs=160)])
    lab_a = []
    plot_runs(ax, a7, "lr", col, D_MARKER[7], ls="-", annotate=list(a7.lr), labels=lab_a)
    plot_runs(ax, a9, "lr", col, D_MARKER[9], ls=":", annotate=list(a9[a9.scored].lr), labels=lab_a)
    setup_log_x(ax, [5e-4, 1e-3, 3e-3, 1e-2], fmt_lr, 3.5e-4, 1.6e-2, "peak learning rate")
    setup_ratio_axis(ax, 0.9, 30)
    mwpm_line(ax)
    ax.legend(handles=[legend_handle(col, D_MARKER[7], "-", "d7 u280 50M, 80 ep"),
                       legend_handle(col, D_MARKER[9], ":", "d9 u280 100M, 160 ep, warmup")],
              loc="upper left", fontsize=7)
    ax.set_title("a  Peak learning rate", loc="left", fontsize=9.5)
    groups += [("a: d7 u280 50M", a7, "plotted"), ("a: d9 u280 100M 160ep", a9, "plotted")]

    # (b) physical error rate p
    ax = axs[0, 1]
    b = select(R, d=7, width="u280", schedule="cosine", lr=0.003, epochs=80, N=1e8, warmup=False, scored=True)
    b = b.sort_values("p")
    lo, hi = wilson(b.k, b.n_shots)
    ax.errorbar(b.p, b.p_L, yerr=[b.p_L - lo, hi - b.p_L], color=col, marker=D_MARKER[7], ms=6,
                mec="white", label="GRU d7 u280")
    mlo, mhi = wilson(np.round(b.mwpm * N_EVAL), N_EVAL)
    ax.errorbar(b.p, b.mwpm, yerr=[b.mwpm - mlo, mhi - b.mwpm], color=MWPM_COLOR, marker="D", ms=4.5,
                ls="--", label="MWPM")
    for r in b.itertuples():
        ax.annotate(f"{r.ratio:.2f}x", (r.p, r.p_L), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8, color="#333333")
    ax.set_yscale("log")
    ax.set_xticks(b.p)
    ax.set_xticklabels([f"{v:g}" for v in b.p])
    ax.set_xlim(0.0027, 0.0053)
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel("p_L (absolute)")
    ax.yaxis.set_major_locator(FixedLocator([0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title("b  Noise level (d7 u280 100M)", loc="left", fontsize=9.5)
    groups.append(("b: noise sweep", b, "plotted"))

    # (c) reset_after True vs False, matched seed and recipe
    ax = axs[1, 0]
    base = R[R.scored & R.pipeline.eq("new") & R.ckpt.eq("val p_L") & R.schedule.eq("constant")
             & np.isclose(R.lr, 0.003) & R.width.eq("u140")]
    pairs = []
    for (d, N, seed), g in base.groupby(["d", "N", "seed"]):
        if set(g.reset_after) == {True, False}:
            f_, t_ = g[~g.reset_after].iloc[0], g[g.reset_after].iloc[0]
            check_one_recipe(pd.DataFrame([f_, t_]).assign(reset_after=False), "reset_after pair")
            pairs.append((d, N, seed, f_, t_))
    for i, (d, N, seed, f_, t_) in enumerate(pairs):
        for j, r in enumerate((f_, t_)):
            lo, hi = wilson(r.k, r.n_shots)
            ax.errorbar(i + (j - 0.5) * 0.36, r.ratio, yerr=[[r.ratio - lo / r.mwpm], [hi / r.mwpm - r.ratio]],
                        fmt=D_MARKER[d], color=WIDTH_COLOR["u140"], ms=7,
                        mfc=WIDTH_COLOR["u140"] if j == 0 else "#1F5F82", mec="white")
            ax.annotate(f"{'False' if j == 0 else 'True'}\n{r.ratio:.3f}x", (i + (j - 0.5) * 0.36, r.ratio),
                        xytext=(-6 if j == 0 else 6, -4 if j == 0 else 4), textcoords="offset points",
                        ha="right" if j == 0 else "left", va="top" if j == 0 else "bottom", fontsize=7,
                        bbox=dict(fc="white", ec="none", pad=0.5), zorder=6)
        ax.plot([i - 0.18, i + 0.18], [f_.ratio, t_.ratio], color="#BBBBBB", lw=1.5, zorder=1)
        groups.append((f"c: d{d} u140 {fmt_shots(N)} s{seed}", pd.DataFrame([f_, t_]), "plotted (matched pair)"))
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([f"d{d} u140 {fmt_shots(N)}\nseed {s}" for d, N, s, _, _ in pairs], fontsize=8)
    ax.set_xlim(-0.75, len(pairs) - 0.25)
    ys = [r.ratio for p in pairs for r in p[3:]]
    setup_ratio_axis(ax, 0.7, max(ys) * 1.35)
    ax.grid(axis="x", visible=False)
    mwpm_line(ax)
    ax.set_title("c  GRU reset_after False vs True", loc="left", fontsize=9.5)

    # (d) schedule length / data at u280 d7
    ax = axs[1, 1]
    dd = pd.concat([
        select(R, d=7, width="u280", schedule="cosine", lr=0.003, epochs=80, N=1e8, p=0.004, warmup=False),
        select(R, d=7, width="u280", schedule="cosine", lr=0.003, epochs=120, N=1e8, p=0.004, warmup=False),
        select(R, d=7, width="u280", schedule="cosine", lr=0.003, epochs=80, N=2e8, p=0.004, warmup=False)])
    seeds50 = select(R, d=7, width="u280", schedule="cosine", lr=0.003, epochs=80, N=5e7, p=0.004,
                     warmup=False, scored=True)
    cv = seeds50.p_L.std(ddof=1) / seeds50.p_L.mean() if len(seeds50) > 1 else np.nan
    m = dd.ratio.mean()
    if np.isfinite(cv):
        ax.axhspan(m * (1 - cv), m * (1 + cv), color="#EEEEEE", lw=0, zorder=0)
        ax.text(-0.45, m * (1 + cv) * 1.002, f"+/-1 seed SD (CV {cv * 100:.1f}%, 50M seed pair)", fontsize=7,
                va="bottom", ha="left", color="#555555")
    for i, r in enumerate(dd.itertuples()):
        lo, hi = wilson(r.k, r.n_shots)
        ax.errorbar(i, r.ratio, yerr=[[r.ratio - lo / r.mwpm], [hi / r.mwpm - r.ratio]], fmt=D_MARKER[7],
                    color=col, ms=7, mec="white")
        ax.annotate(f"{r.ratio:.3f}x", (i, r.ratio), xytext=(8, 0), textcoords="offset points",
                    fontsize=8, va="center", color="#333333")
    ax.set_xticks(range(len(dd)))
    ax.set_xticklabels([f"{r.data}\n{r.epochs} ep" for r in dd.itertuples()], fontsize=8)
    ax.set_xlim(-0.5, len(dd) - 0.3)
    ax.set_yscale("log")
    ax.set_ylim(1.0, 1.13)
    ax.yaxis.set_major_locator(FixedLocator([1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.set_ylabel("p_L / MWPM")
    ax.grid(axis="x", visible=False)
    ax.set_title("d  Epochs and data at d7 u280", loc="left", fontsize=9.5)
    groups.append(("d: schedule/data", dd, "plotted"))
    groups.append(("d: seed band (50M pair)", seeds50, "reference"))

    spread = dd.ratio.max() - dd.ratio.min()
    title = (f"Peak LR moves p_L most; 80 vs 120 ep and 100M vs 200M at d7 u280 span {spread:.3f}x "
             f"(within noise)")
    pend_a = P[P.d.eq(9) & np.isclose(P.N, 1e8) & ~P.val_only].run_id.tolist() if len(P) else []
    cap = ("(a) Cosine, clip 0.64; x = diverged or killed. (b) Same run recipe at each p. (c) Matched seed and "
           "recipe only. (d) Band = seed-to-seed SD at 50M. "
           + (f"Pending: {', '.join(pend_a)}. " if pend_a else "") + common_caption(data))
    bb = finish(fig, title, cap)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.93, bottom=bb)
    place_labels(axs[0, 0], lab_a)
    return FigResult("fig5_comparisons", fig, title, cap, runs_used_table("fig5", groups))


# --------------------------------------------------------------------------------------
# Figure 6: seed noise
# --------------------------------------------------------------------------------------
def fig6_seed_noise(data: Data) -> FigResult:
    R = data.runs
    base = R[R.scored & R.stability.eq("stable")]
    rows, groups = [], []
    for key, g in base.groupby(["d", "width", "N"] + RECIPE_KEYS):
        if g.seed.nunique() < 2:
            continue
        g = g.drop_duplicates("seed")
        check_one_recipe(g, "seed group")
        mean, sd = g.p_L.mean(), g.p_L.std(ddof=1)
        rows.append(dict(d=key[0], width=key[1], N=key[2], n_seeds=len(g), mean_ratio=g.ratio.mean(),
                         cv_pct=100 * sd / mean, mwpm=g.mwpm.iloc[0], recipe=g.recipe.iloc[0],
                         seeds=",".join(map(str, sorted(g.seed))),
                         runs="; ".join(f"line {int(r.csv_line)}" for r in g.itertuples())))
        groups.append((f"d{key[0]} {key[1]} {fmt_shots(key[2])}", g, "seed group"))
    T = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    # expected CV from shot noise alone: 1/sqrt(n p)
    xx = np.logspace(np.log10(0.7), np.log10(30), 100)
    for d in sorted(T.d.unique()):
        cvb = 100 / np.sqrt(N_EVAL * xx * data.mwpm[d])
        ax.plot(xx, cvb, color="#AAAAAA", lw=1, ls=":", zorder=1)
        ax.annotate(f"shot noise only, d{d}", (xx[-12], cvb[-12]), xytext=(0, 2), textcoords="offset points",
                    fontsize=7, ha="right", va="bottom", color="#777777", rotation=-8)
    lab6 = []
    for r in T.itertuples():
        ls_old = "older pipeline" in r.recipe or "old pipeline" in r.recipe
        ax.plot(r.mean_ratio, r.cv_pct, marker=D_MARKER[r.d], color=WIDTH_COLOR[r.width], ms=8,
                mec="black" if ls_old else "white", mew=0.7, ls="none", zorder=3)
        tag = "const" if "constant" in r.recipe else "cos"
        ra = ", ra=T" if "reset_after=True" in r.recipe else ""
        lab6.append((r.mean_ratio, r.cv_pct, f"d{r.d} {r.width} {fmt_shots(r.N)} {tag}{ra}, n={r.n_seeds}"))
    setup_log_x(ax, [0.8, 1, 1.5, 2, 3, 5, 10, 20], lambda v: f"{v:g}x", 0.75, 30, "mean p_L / MWPM over seeds")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.3, 0.5, 1, 2, 3, 5, 10]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_ylim(0.15, 12)
    ax.set_ylabel("seed-to-seed CV of p_L (%)")
    ax.axvline(1, color="black", ls="--", lw=1)
    hs = [legend_handle(c, "o", "", w) for w, c in WIDTH_COLOR.items() if w in set(T.width)]
    hs += [Line2D([], [], marker=D_MARKER[d], color="#888888", ls="", label=f"d = {d}") for d in sorted(T.d.unique())]
    hs += [Line2D([], [], marker="o", color="#888888", mec="black", ls="", label="older pipeline (black edge)")]
    ax.legend(handles=hs, loc="lower left", fontsize=7, ncol=3, bbox_to_anchor=(0.0, 1.0))
    near, far = T[T.mean_ratio < 1.3], T[T.mean_ratio >= 1.5]
    title = (f"Seed-to-seed CV of p_L: median {far.cv_pct.median():.1f}% for configs at >= 1.5x MWPM, "
             f"{near.cv_pct.median():.1f}% within 1.3x of MWPM")
    cap = ("One point per config with >= 2 seeds and one recipe. Dotted = CV expected from eval shot noise "
           "alone. " + common_caption(data))
    b = finish(fig, title, cap)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=b)
    place_labels(ax, lab6, fontsize=7)
    res = FigResult("fig6_seed_noise", fig, title, cap, runs_used_table("fig6", groups))
    res.table = T.assign(mean_ratio=T.mean_ratio.round(3), cv_pct=T.cv_pct.round(2),
                         N=T.N.map(fmt_shots))
    return res


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
FIGURES = [fig1_data_scaling, fig2_model_size, fig3_distance, fig4_generalization, fig5_comparisons,
           fig6_seed_noise]


def make_all(data: Data | None = None, out_dir: Path = OUT_DIR, save=True) -> list[FigResult]:
    data = data or load_all()
    results = [f(data) for f in FIGURES]
    if not save:
        return results
    out_dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_dir / "scaling_summary.pdf") as pdf:
        for r in results:
            r.fig.savefig(out_dir / f"{r.key}.pdf")
            r.fig.savefig(out_dir / f"{r.key}.png", dpi=300)
            pdf.savefig(r.fig)
    fits = [f.row() for r in results for f in r.fits]
    ft = pd.DataFrame(fits)
    ft.to_csv(out_dir / "fit_parameters.csv", index=False)
    lam = next(r.table for r in results if r.key == "fig3_distance")
    lam.to_csv(out_dir / "lambda_table.csv", index=False)
    next(r.table for r in results if r.key == "fig6_seed_noise").to_csv(out_dir / "seed_noise_table.csv", index=False)
    pd.concat([r.runs_used for r in results], ignore_index=True).to_csv(out_dir / "runs_used.csv", index=False)
    write_fit_markdown(ft, lam, data, out_dir / "fit_parameters.md")
    return results


def write_fit_markdown(ft: pd.DataFrame, lam: pd.DataFrame, data: Data, path: Path):
    L = [f"# Fit parameters", "", f"Generated by figures/scaling_figs.py from {data.source}. "
         f"95% CIs from {N_BOOT} bootstrap replicates (seeds resampled at each x, error counts redrawn).", ""]
    L += ["## Power-law fits, p_L = E + B x^(-beta)", "",
          "| figure | series | E (x MWPM) [95% CI] | beta [95% CI] | runs / distinct x | converged |",
          "|---|---|---|---|---|---|"]
    for r in ft.itertuples():
        L.append(f"| {r.figure} | {r.series} | {fmt_ci(r.E_over_MWPM, r.E_over_MWPM_lo, r.E_over_MWPM_hi, 3)} | "
                 f"{fmt_ci(r.beta, r.beta_lo, r.beta_hi)} | {r.n_runs} / {r.n_distinct_x} | {r.bootstrap_fits_converged} |")
    L += ["", "### Runs in each fit", ""]
    for r in ft.itertuples():
        L += [f"**{r.series}** ({r.model}). Recipe: {r.recipe}.", ""]
        L += [f"- {x.strip()}" for x in r.runs.split(";")] + [""]
    L += ["## Lambda = p_L(d) / p_L(d+2)", "", "| series | d -> d+2 | Lambda [95% CI] |", "|---|---|---|"]
    for r in lam.itertuples():
        L.append(f"| {r.series} | {r.from_d} -> {r.to_d} | {fmt_ci(r.Lambda, r.Lambda_lo, r.Lambda_hi)} |")
    path.write_text("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", help="all-runs CSV (default: newest GRU_all_runs_*.csv in the repo root)")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--extra-root", action="append", default=[],
                    help="another checkout to search for pending runs' run.log files (repeatable)")
    a = ap.parse_args(argv)
    data = load_all(a.csv, [Path(p) for p in a.extra_root])
    res = make_all(data, Path(a.out))
    for r in res:
        print(f"{r.key}: {r.title}")
    print(f"wrote {Path(a.out)}")


if __name__ == "__main__":
    main()
