#!/usr/bin/env python3
# Created: 2026-10-12
# Last updated: 2026-10-12

"""Results table for animateQEC's Results scene: MWPM and each named GRU run on the same block.

Error counts come only from an error-count column, never back-derived from p_L or the ratio.

python scripts/export_results.py --runs GRU_all_runs_2026-10-01.csv --probes f2280 f3280 \
    --block-label "preliminary · eval block, not the held-out test set" \
    --out ../animateQEC/examples/gru-decoder/data/results_eval_preliminary.csv
"""
import argparse
import csv
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--probes", nargs="+", required=True, help="run names as written 'probe <name>' in the notes")
ap.add_argument("--block", default="eval")
ap.add_argument("--block-label", required=True)
ap.add_argument("--errors-column", default="n_errors", help="GRU error count column, if the CSV has one")
ap.add_argument("--mwpm-errors-column", default="mwpm_n_errors")
ap.add_argument("--n-shots-column", default="n_shots")
ap.add_argument("--out", required=True)
args = ap.parse_args()

rows = list(csv.DictReader(open(args.runs)))
NOTES = "Notes" if "Notes" in rows[0] else "Checkpoint criterion"


def find(probe):
    hits = [r for r in rows if re.search(rf"\bprobe {re.escape(probe)}(?![\w-])", r[NOTES])]
    assert len(hits) == 1, f"probe {probe}: {len(hits)} rows"
    return hits[0]


def count(r, col):
    v = r.get(col, "")
    return int(v) if v not in ("", None) else None


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


out = []
mwpm_ler = None
for probe in args.probes:
    r = find(probe)
    m = re.search(r"McNemar vs MWPM: b=(\d+) c=(\d+) n_disc=(\d+) p=([0-9.e+-]+)", r[NOTES])
    n, k = count(r, args.n_shots_column), count(r, args.errors_column)
    ci = clopper_pearson(k, n) if (n and k is not None) else (None, None)
    if mwpm_ler is None:
        mwpm_ler, mwpm_row = float(r["MWPM"]), r
    assert float(r["MWPM"]) == mwpm_ler, f"{probe}: MWPM p_L differs, not the same block"
    out.append({"label": probe, "kind": "gru", "model": r["Width"], "block": args.block,
                "n_shots": n, "n_errors": k, "ler": float(r["p_L"]), "ci_low": ci[0], "ci_high": ci[1],
                "ratio_to_mwpm": float(r["Relative to MWPM"]),
                "mcnemar_b": int(m[1]) if m else None, "mcnemar_c": int(m[2]) if m else None,
                "mcnemar_p": m[4] if m else None, "block_label": args.block_label,
                "source": os.path.basename(args.runs)})

n, k = count(mwpm_row, args.n_shots_column), count(mwpm_row, args.mwpm_errors_column)
ci = clopper_pearson(k, n) if (n and k is not None) else (None, None)
out.insert(0, {"label": "MWPM", "kind": "mwpm", "model": "pymatching", "block": args.block,
               "n_shots": n, "n_errors": k, "ler": mwpm_ler, "ci_low": ci[0], "ci_high": ci[1],
               "ratio_to_mwpm": 1.0, "mcnemar_b": None, "mcnemar_c": None, "mcnemar_p": None,
               "block_label": args.block_label, "source": os.path.basename(args.runs)})

with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows({k: ("" if v is None else v) for k, v in row.items()} for row in out)
for row in out:
    print({k: row[k] for k in ("label", "ler", "n_errors", "ci_low", "ratio_to_mwpm", "mcnemar_b",
                               "mcnemar_c", "mcnemar_p")})
print("wrote", args.out)
