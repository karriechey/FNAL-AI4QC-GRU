#!/usr/bin/env python3
# Created: 2026-10-12
# Last updated: 2026-10-12 (no file paths in the output: checkpoint by file name, code copy by folder name; checked before writing)

"""Trace evaluation-block shots through a stacked-GRU checkpoint for animateQEC's RealShot.

Per shot: detection events, hidden states, z and r gates, P(flip), integrated gradients, and
MWPM's prediction and edges; plus a per-layer 2-D PCA. Refuses shots outside the evaluation block.

~/tfenv/bin/python export_shot_trace.py --src ~/FNAL-AI4QC-GRU-fresh/src \
    --weights <ckpt>/...best.weights.h5 --units 280 280 280 \
    --shots 16421787 16612857 --out ~/shot_trace.json
"""
import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="src/ of the code copy that trained the checkpoint")
ap.add_argument("--weights", required=True, help=".best.weights.h5")
ap.add_argument("--units", type=int, nargs="+", required=True)
ap.add_argument("--shots", type=int, nargs="+", required=True,
                help="pool shot indices; the first is the main shot, the rest backups")
ap.add_argument("--d", type=int, default=9)
ap.add_argument("--p", type=float, default=0.004)
ap.add_argument("--n-pca", type=int, default=2000, help="evaluation-block shots the PCA is fit on")
ap.add_argument("--ig-steps", type=int, default=64)
ap.add_argument("--out", required=True)
args = ap.parse_args()

src = os.path.abspath(os.path.expanduser(args.src))
sys.path.insert(0, src)
import partitions
import pool_rows
import tensorflow as tf
from circuit_generators import build_circuit
from StudentModels import build_gru_student, detector_sequence_layout, to_sequence
from tensorflow.keras import layers as KL

d, r, p = args.d, args.d, args.p
pool = partitions.pool_path(d, "formal")
a, b = partitions.block("eval")
s0, s1 = partitions.block("sealed")
assert b <= s0 or a >= s1, "evaluation block overlaps the sealed block"
for s in args.shots:
    assert a <= s < b, f"shot {s} is outside the evaluation block [{a}, {b})"


def read(i0, i1):
    return (pool_rows.rows(pool, "det_evts", i0, i1).astype(np.uint8),
            pool_rows.rows(pool, "flips", i0, i1).reshape(-1).astype(bool))


# ---- model -----------------------------------------------------------------------------
model = build_gru_student(d=d, rounds=r, inputs="evts", units=tuple(args.units), hidden=(), p=p,
                          reset_after=False, unroll=False)
model.load_weights(args.weights)
grus = [l for l in model.layers if isinstance(l, KL.GRU)]
assert len(grus) == len(args.units)

# every layer's output sequence; the last layer is cloned with return_sequences=True
x_in = KL.Input(shape=model.input_shape[1:])
h, seqs = x_in, []
for i, g in enumerate(grus):
    if g.return_sequences:
        h = g(h)
    else:
        cfg = dict(g.get_config(), return_sequences=True, name=g.name + "_seq")
        clone = KL.GRU.from_config(cfg)
        h = clone(h)
        clone.set_weights(g.get_weights())
    seqs.append(h)
logit_layer = model.get_layer("logit")
trace_model = tf.keras.Model(x_in, [*seqs, logit_layer(seqs[-1][:, -1])])


def run(x):
    out = trace_model.predict(x, batch_size=1000, verbose=0)
    return out[:-1], out[-1].reshape(-1)


def gates(layer, xs):
    # z, r, h of a reset_after=False GRU recomputed in numpy
    assert not layer.reset_after
    W, U, bias = layer.get_weights()
    u = layer.units
    sig = {"sigmoid": lambda v: 1 / (1 + np.exp(-v)),
           "hard_sigmoid": lambda v: np.clip(0.2 * v + 0.5, 0, 1)}[layer.recurrent_activation.__name__]
    assert layer.activation.__name__ == "tanh"
    hp = np.zeros(u)
    zs, rs, hs = [], [], []
    for xt in xs:
        xw = xt @ W + bias
        z = sig(xw[:u] + hp @ U[:, :u])
        rr = sig(xw[u:2 * u] + hp @ U[:, u:2 * u])
        hh = np.tanh(xw[2 * u:] + (rr * hp) @ U[:, 2 * u:])
        hp = z * hp + (1 - z) * hh
        zs.append(z), rs.append(rr), hs.append(hp)
    return np.array(zs), np.array(rs), np.array(hs)


# ---- MWPM ------------------------------------------------------------------------------
import pymatching
circ = build_circuit(d, p, r)
matcher = pymatching.Matching.from_detector_error_model(circ.detector_error_model(decompose_errors=True))
_, _, index_map = detector_sequence_layout(d, r, p)
t_of = np.full(circ.num_detectors, -1)
for t, row in enumerate(index_map):
    t_of[row[row >= 0]] = t

# ---- traces ----------------------------------------------------------------------------
traces = []
for k, s in enumerate(args.shots):
    det, flip = read(s, s + 1)
    x = to_sequence(det, d, r, p)
    hidden, logit = run(x)
    p_model = float(tf.sigmoid(model.predict(x, verbose=0)).numpy().reshape(-1)[0])
    p_trace = float(1 / (1 + np.exp(-logit[0])))
    assert abs(p_trace - p_model) < 1e-6, f"shot {s}: trace P(flip) {p_trace} != model {p_model}"

    layer_in, layer_gates = x[0], []
    for i, g in enumerate(grus):
        z, rr, hh = gates(g, layer_in)
        err = float(np.abs(hh - hidden[i][0]).max())
        assert err < 1e-5, f"shot {s} layer {i}: recomputed h differs from Keras by {err}"
        layer_gates.append({"z": np.round(z, 6).tolist(), "r": np.round(rr, 6).tolist(),
                            "max_abs_err_vs_keras": err})
        layer_in = hidden[i][0]

    # integrated gradients of the logit, baseline all zeros
    xt = tf.constant(x, tf.float32)
    alphas = (np.arange(args.ig_steps) + 0.5) / args.ig_steps
    path = tf.constant(alphas[:, None, None] * x, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(path)
        out = model(path, training=False)
    grad = tape.gradient(out, path).numpy().mean(0)
    attr_map = (x[0] * grad)
    attribution = {str(int(index_map[t, j])): float(attr_map[t, j])
                   for t, j in zip(*np.nonzero(x[0])) if index_map[t, j] >= 0}

    fired = np.flatnonzero(det[0])
    by_round = {}
    for i in fired:
        by_round.setdefault(str(int(t_of[i])), []).append(int(i))
    mwpm_pred = bool(matcher.decode(det[0])[0])
    edges = matcher.decode_to_edges_array(det[0]).astype(int).tolist()

    traces.append({
        "role": "main" if k == 0 else "backup",
        "pool_shot": int(s), "block": "eval", "block_index": int(s - a),
        "n_events": int(len(fired)), "events_by_round": by_round,
        "truth": bool(flip[0]), "p_flip": p_model, "logit": float(logit[0]),
        "gru_pred": p_model > 0.5, "mwpm_pred": mwpm_pred, "mwpm_edges": edges,
        "hidden": [np.round(hidden[i][0], 6).tolist() for i in range(len(grus))],
        "gates": layer_gates,
        "attribution": {"method": "integrated gradients of the logit, baseline all zeros",
                        "steps": args.ig_steps, "per_detector": attribution},
    })
    print(f"shot {s}: {len(fired)} events, truth {int(flip[0])}, MWPM {int(mwpm_pred)}, "
          f"GRU P(flip) {p_model:.6f} (trace {p_trace:.6f}), gates max err "
          f"{max(g['max_abs_err_vs_keras'] for g in layer_gates):.2e}")

# ---- PCA on the first n_pca evaluation-block shots (traced shots are not among them)
det_c, flip_c = read(a, a + args.n_pca)
assert not set(args.shots) & set(range(a, a + args.n_pca))
hid_c, logit_c = run(to_sequence(det_c, d, r, p))
pca = []
for i, hc in enumerate(hid_c):
    states = hc.reshape(-1, hc.shape[-1])  # every step of every shot
    mean = states.mean(0)
    _, sv, vt = np.linalg.svd(states - mean, full_matrices=False)
    var = sv ** 2 / (sv ** 2).sum()
    comp = vt[:2]
    proj = lambda v: np.round((np.asarray(v) - mean) @ comp.T, 5).tolist()
    pca.append({
        "layer": i + 1, "explained_variance_ratio": [float(v) for v in var[:2]],
        "fit_on": f"all {r + 1} steps of {args.n_pca} shots, evaluation block [{a}, {a + args.n_pca})",
        "cloud_final": proj(hc[:, -1]), "cloud_truth": flip_c.tolist(),
        "trajectories": {str(t["pool_shot"]): proj(t["hidden"][i]) for t in traces},
    })
    print(f"PCA layer {i + 1}: explained variance PC1 {var[0]:.3f}, PC2 {var[1]:.3f}")

with open(args.weights, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()
best_json = args.weights.replace(".weights.h5", ".json")
out = {
    "meta": {
        "checkpoint": os.path.basename(args.weights), "sha256": sha,
        "best": json.load(open(best_json)) if os.path.exists(best_json) else None,
        "units": args.units, "d": d, "rounds": r, "p": p, "pool": "FORMAL",
        "block": "eval", "block_range": [a, b], "held_out_test_set": False,
        "code_copy": os.path.basename(os.path.dirname(src)), "tf_version": tf.__version__,
        "device": "CPU", "host": platform.node(), "script": "scripts/export_shot_trace.py",
    },
    "shots": traces,
    "pca": pca,
}

def assert_no_paths(obj, where="out"):
    # output is published: refuse anything that looks like a file path
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
    json.dump(out, f)
print(f"wrote {args.out} ({os.path.getsize(os.path.expanduser(args.out)) / 1e6:.2f} MB)")
