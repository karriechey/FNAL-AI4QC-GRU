#!/usr/bin/env python3
# Created: 2026-10-12
# Last updated: 2026-10-12
"""Write a stacked-GRU checkpoint's architecture (model.summary() and per-layer config) to JSON.

For animateQEC's architecture diagrams and parameter labels. Builds the model exactly as
eval_student_on_tail.py does, loads the weights (so the config is proven to fit the
checkpoint), and records each layer's class, units, activations, reset_after, shapes and
parameter count. Also checks that logits_to_prob is the logistic sigmoid. No file paths are
written. Run on grace1 in ~/tfenv:

    ~/tfenv/bin/python export_model_config.py --src ~/FNAL-AI4QC-GRU-fresh/src --run f3280 \\
        --weights <ckpt>/...best.weights.h5 --units 280 280 280 --out ~/model_f3280.json
"""
import argparse
import hashlib
import io
import json
import os
import sys
from contextlib import redirect_stdout

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--run", required=True, help="run name, e.g. f3280")
ap.add_argument("--weights", required=True)
ap.add_argument("--units", type=int, nargs="+", required=True)
ap.add_argument("--d", type=int, default=9)
ap.add_argument("--p", type=float, default=0.004)
ap.add_argument("--out", required=True)
args = ap.parse_args()

sys.path.insert(0, os.path.abspath(os.path.expanduser(args.src)))
import tensorflow as tf
from StudentModels import build_gru_student, logits_to_prob

model = build_gru_student(d=args.d, rounds=args.d, inputs="evts", units=tuple(args.units), hidden=(),
                          p=args.p, reset_after=False, unroll=False)
model.load_weights(args.weights)

z = np.array([-3.0, 0.0, 2.0])
assert np.allclose(np.asarray(logits_to_prob(z)).reshape(-1), 1 / (1 + np.exp(-z)), atol=1e-6), \
    "logits_to_prob is not the logistic sigmoid"


def shape(s):
    return [None if v is None else int(v) for v in s]


layers = []
for l in model.layers:
    cfg = l.get_config()
    entry = {"name": l.name, "class": type(l).__name__,
             "input_shape": shape(l.input_shape[0] if isinstance(l.input_shape, list) else l.input_shape),
             "output_shape": shape(l.output_shape[0] if isinstance(l.output_shape, list) else l.output_shape),
             "params": int(l.count_params())}
    for k in ("units", "return_sequences", "reset_after", "activation", "recurrent_activation",
              "use_bias"):
        if k in cfg:
            entry[k] = cfg[k]
    layers.append(entry)

buf = io.StringIO()
with redirect_stdout(buf):
    model.summary(line_length=100)
with open(args.weights, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()
best_json = args.weights.replace(".weights.h5", ".json")
out = {
    "run": args.run, "checkpoint": os.path.basename(args.weights), "sha256": sha,
    "best": json.load(open(best_json)) if os.path.exists(best_json) else None,
    "d": args.d, "rounds": args.d, "p": args.p, "tf_version": tf.__version__,
    "input_shape": shape(model.input_shape), "total_params": int(model.count_params()),
    "output": "logit; P(flip) = logits_to_prob(logit) = sigmoid(logit), outside the model",
    "layers": layers, "summary": buf.getvalue().splitlines(),
}


def assert_no_paths(obj, where="out"):
    """Published JSON: refuse any string that looks like a file-system path."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert_no_paths(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert_no_paths(v, f"{where}[{i}]")
    elif isinstance(obj, str):
        assert not (obj.startswith(("/", "~")) or "/home/" in obj or "/Users/" in obj
                    or os.path.expanduser("~") in obj), f"path in {where}: {obj!r}"


assert_no_paths(out)
with open(os.path.expanduser(args.out), "w") as f:
    json.dump(out, f, indent=1)
print("\n".join(out["summary"]))
for l in layers:
    print(l)
print("wrote", args.out)
