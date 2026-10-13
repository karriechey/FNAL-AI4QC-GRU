# Vendored from animateQEC animateqec/stim_io.py @ b436ed3
"""Stim circuit -> plain dict that the scenes draw from."""
from collections import defaultdict

import numpy as np
import stim

TWO_QUBIT = {"CX", "CNOT", "CZ", "CY", "XCZ", "YCZ", "ZCX", "ZCY", "ZCZ"}
MEASURE = {"M", "MZ", "MR", "MRZ", "MX", "MRX", "MY", "MRY"}


def _first_round(circuit):
    """Ancillas, X-type ancillas and CX partners from the first measured round."""
    hadamard, partners, seen_cx = set(), defaultdict(set), False
    for inst in circuit.flattened():
        name = inst.name
        qs = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        if name == "H":
            hadamard.update(qs)
        elif name in TWO_QUBIT:
            seen_cx = True
            for a, b in zip(qs[::2], qs[1::2]):
                partners[a].add(b)
                partners[b].add(a)
        elif name in MEASURE and seen_cx:
            return qs, hadamard, partners
    raise ValueError("no measurement after two-qubit gates")


def circuit_structure(circuit: stim.Circuit) -> dict:
    """Qubits, stabilizers, detectors and rounds of a memory-type circuit."""
    coords = circuit.get_final_qubit_coordinates()
    ancillas, hadamard, partners = _first_round(circuit)
    anc_set = set(ancillas)
    data = sorted(q for q in coords if q not in anc_set)
    xy = lambda q: [float(c) for c in coords[q][:2]]
    basis = {a: "X" if a in hadamard else "Z" for a in ancillas}

    stabilizers = []
    for a in ancillas:
        support = sorted(q for q in partners[a] if q not in anc_set)
        stabilizers.append({"ancilla": a, "xy": xy(a), "basis": basis[a],
                            "data": support, "data_xy": [xy(q) for q in support]})

    anc_at = {tuple(xy(a)): a for a in ancillas}
    detectors, rounds = [], defaultdict(list)
    for i, c in sorted(circuit.get_detector_coordinates().items()):
        x, y, t = float(c[0]), float(c[1]), int(c[2])
        a = anc_at.get((x, y))
        detectors.append({"index": i, "x": x, "y": y, "t": t,
                          "basis": basis.get(a), "ancilla": a})
        rounds[t].append(i)

    return {
        "qubits": {"data": [{"q": q, "xy": xy(q)} for q in data],
                   "ancilla": [{"q": a, "xy": xy(a), "basis": basis[a]} for a in ancillas]},
        "stabilizers": stabilizers,
        "detectors": detectors,
        "rounds": {str(t): v for t, v in sorted(rounds.items())},
    }


def sample_shots(circuit, n, seed, n_readable=5, max_events=None, mwpm=True,
                 n_search=0, search_max_events=25):
    """A few readable shots, plus MWPM failures with few events if pymatching is installed.

    Shots are drawn in chunks of n from one seeded sampler, so shot indices are stable:
    the first n shots are the same whether or not the n_search extension runs.
    """
    sampler = circuit.compile_detector_sampler(seed=seed)
    matcher = None
    if mwpm:
        try:
            import pymatching
            matcher = pymatching.Matching.from_detector_error_model(
                circuit.detector_error_model(decompose_errors=True))
        except ImportError:
            pass
    t_of = np.array([int(c[2]) for _, c in sorted(circuit.get_detector_coordinates().items())])

    def shot(i, det_row, flip, wrong, pick):
        by_round = defaultdict(list)
        for d in np.flatnonzero(det_row):
            by_round[str(t_of[d])].append(int(d))
        return {"shot": int(i), "pick": pick, "n_events": int(det_row.sum()),
                "events_by_round": dict(by_round), "logical_flip": bool(flip),
                "mwpm_wrong": wrong}

    def draw():
        det, obs = sampler.sample(n, separate_observables=True)
        flips = obs[:, 0].astype(bool)
        wrong = None if matcher is None else matcher.decode_batch(det)[:, 0].astype(bool) != flips
        return det, flips, wrong

    det, flips, wrong = draw()
    n_events = det.sum(1)
    readable = np.flatnonzero((n_events >= 4) & (n_events <= (max_events or np.inf)))
    readable = readable[np.argsort(n_events[readable], kind="stable")]
    out = [shot(i, det[i], flips[i], None if wrong is None else bool(wrong[i]), "readable")
           for i in readable[:n_readable]]
    if wrong is None:
        return out

    best = None  # (n_events, global index, det row, flip) of the fewest-event MWPM failure
    for offset in range(0, max(n, n_search), n):
        if offset:
            det, flips, wrong = draw()
        for i in np.flatnonzero(wrong):
            k = int(det[i].sum())
            if best is None or k < best[0]:
                best = (k, offset + i, det[i].copy(), flips[i])
        if offset == 0 and best is not None:
            out.append(shot(best[1], best[2], best[3], True, "mwpm_fail"))
    if n_search > n and best is not None and best[0] < search_max_events \
            and best[1] != out[-1]["shot"]:
        out.append(shot(best[1], best[2], best[3], True, f"mwpm_fail_search_{n_search}"))
    return out


def detslice_svg(circuit, x, y, t):
    """Detslice diagram of one detector, as SVG text."""
    return str(circuit.diagram("detslice-svg", filter_coords=[(x, y, t)]))


def build_viz(circuit, meta=None, n_samples=20_000, seed=0, **sample_kw):
    """Structure + samples + meta, ready to dump as viz_data.json."""
    viz = circuit_structure(circuit)
    per_round = [len(v) for v in viz["rounds"].values()]
    viz["meta"] = {
        "num_detectors": circuit.num_detectors,
        "detectors_per_round": per_round,
        "stim_version": stim.__version__,
        "seed": seed,
        **(meta or {}),
    }
    viz["samples"] = sample_shots(circuit, n_samples, seed, **sample_kw)
    viz["results"] = []
    return viz
