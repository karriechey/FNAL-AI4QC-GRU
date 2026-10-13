#!/usr/bin/env python3
# Created: 2026-10-12
# Last updated: 2026-10-12 (exports Stim detecting regions of the central Z and X detector)
"""Export the training circuit's lattice, detectors and sample shots to viz_data.json for animateQEC."""
import argparse
import csv
import json
import os
import subprocess
import sys

import stim

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import stim_io  # vendored from animateQEC, next to this script
from circuit_generators import build_circuit  # the circuit the decoder trains on

ap = argparse.ArgumentParser()
ap.add_argument("--circuit", help="path to a .stim file; default is build_circuit(d, p, rounds=d)")
ap.add_argument("--distance", type=int, default=9)
ap.add_argument("--p", type=float, default=0.004)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--n-search", type=int, default=1_000_000,
                help="shots searched for the fewest-event MWPM failure (kept if < 25 events)")
ap.add_argument("--results", help="CSV with label, ler, ci_low, ci_high, kind")
ap.add_argument("--out", default="viz_data.json")
args = ap.parse_args()

if args.circuit:
    circuit, source = stim.Circuit.from_file(args.circuit), args.circuit
    d = r = None
else:
    d = r = args.distance
    circuit = build_circuit(d, args.p, rounds=r)
    source = f"circuit_generators.build_circuit(d={d}, p={args.p}, rounds={r})"

commit = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
meta = {"source": source, "distance": d, "rounds": r, "p": args.p, "basis": "Z",
        "git_commit": commit}
structure = stim_io.circuit_structure(circuit)
centre = {b: stim_io.central_detector(structure, b, 1) for b in ("Z", "X")}
viz = stim_io.build_viz(circuit, meta, seed=args.seed, max_events=3 * (d or 9),
                        n_search=args.n_search, region_detectors=list(centre.values()))
# build_viz puts its own keys first; keep meta's order readable
viz = {"meta": viz.pop("meta"), **viz}

if args.results:
    with open(args.results) as f:
        viz["results"] = [{k: (float(v) if k in ("ler", "ci_low", "ci_high") and v else v)
                           for k, v in row.items()} for row in csv.DictReader(f)]

out_dir = os.path.dirname(os.path.abspath(args.out))
os.makedirs(os.path.join(out_dir, "svg"), exist_ok=True)
for b, i in centre.items():
    c = viz["detectors"][i]
    with open(os.path.join(out_dir, "svg", f"detector_{b}_t1.svg"), "w") as f:
        f.write(stim_io.detslice_svg(circuit, c["x"], c["y"], 1))
    viz["meta"][f"detslice_{b}"] = i

with open(args.out, "w") as f:
    json.dump(viz, f, indent=1)

m = viz["meta"]
print("source:", m["source"], "| stim", m["stim_version"], "| commit", commit)
print("data qubits:", len(viz["qubits"]["data"]), "| ancillas:", len(viz["qubits"]["ancilla"]),
      "(Z", sum(a["basis"] == "Z" for a in viz["qubits"]["ancilla"]),
      "X", sum(a["basis"] == "X" for a in viz["qubits"]["ancilla"]), ")")
print("detectors:", m["num_detectors"], "| per round:", m["detectors_per_round"])
w = [len(s["data"]) for s in viz["stabilizers"]]
print("stabilizer weights:", {k: w.count(k) for k in sorted(set(w))})
for s in viz["samples"]:
    print(f"  {s['pick']:<26} shot {s['shot']:>6}: {s['n_events']:>2} events, rounds",
          {t: len(v) for t, v in s["events_by_round"].items()},
          "flip", s["logical_flip"], "mwpm_wrong", s["mwpm_wrong"])
for i, reg in viz["regions"].items():
    print(f"  region of detector {i}: ticks {reg[0]['tick']}-{reg[-1]['tick']},",
          "qubits per tick", [len(r["qubits"]) for r in reg])
