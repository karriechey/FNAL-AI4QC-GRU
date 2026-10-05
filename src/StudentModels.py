#!/usr/bin/env python3
# from FNAL-QCDecoding-FPGA/StudentModels.py @ d5ee86f
# Created: 2026-07-29
# Last modified: 2026-10-05 (units may be a sequence: stacked GRU layers; a single int builds exactly the model as before). Earlier 2026-09-26 (reset_after and unroll are build arguments, default False)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras import Model


def _patch_qkeras_recurrent_nest():
    """Restore tf.python.util.nest.is_sequence for QKeras 0.9's recurrent cells.

    QKeras 0.9's QGRUCell/QLSTMCell call `nest.is_sequence(states)`, which TensorFlow
    renamed to `nest.is_nested`, so every QGRU/QLSTM forward pass dies on the pinned
    TF 2.15. Same predicate under two names, so aliasing it back is faithful. Installed
    only when the attribute is absent. Candidate upstream patch.
    """
    try:
        from tensorflow.python.util import nest as _nest
    except ImportError:  # pragma: no cover - TF internals moved; let QKeras fail loudly
        return
    if not hasattr(_nest, 'is_sequence') and hasattr(_nest, 'is_nested'):
        _nest.is_sequence = _nest.is_nested


_patch_qkeras_recurrent_nest()


_LAYOUT_CACHE = {}


def detector_sequence_layout(d, rounds, p):
    """Map the flat det_evts vector onto (timestep, plaquette position).

    The short timesteps' positions are subsets of the full 24, so each timestep fits a
    fixed 24-wide slot vector, zero where nothing was measured. Column j then means the
    same plaquette at every timestep, and the input stays rectangular for hls4ml.

    Returns (n_timesteps, n_positions, index_map); index_map[t, j] is the detector index
    for position j at timestep t, or -1 if unmeasured.
    """
    key = (d, rounds, round(p, 6))
    if key in _LAYOUT_CACHE:
        return _LAYOUT_CACHE[key]

    from circuit_generators import build_circuit
    circ = build_circuit(d, p, rounds)
    coords = circ.get_detector_coordinates()
    n_det = circ.num_detectors

    xy = [(coords[i][0], coords[i][1]) for i in range(n_det)]
    ts = [int(coords[i][2]) for i in range(n_det)]
    positions = sorted(set(xy))
    pos_index = {q: j for j, q in enumerate(positions)}
    n_t, n_pos = max(ts) + 1, len(positions)

    index_map = np.full((n_t, n_pos), -1, dtype=np.int32)
    for i in range(n_det):
        index_map[ts[i], pos_index[xy[i]]] = i

    _LAYOUT_CACHE[key] = (n_t, n_pos, index_map)
    return _LAYOUT_CACHE[key]


def to_sequence(det_evts, d, rounds, p):
    """Scatter flat det_evts [N, n_det] into [N, n_timesteps, n_positions], zero-filled.

    In numpy rather than a Lambda/gather inside the model: the reordering is fixed input
    wiring, free on FPGA, and hls4ml never sees a gather.
    """
    n_t, n_pos, index_map = detector_sequence_layout(d, rounds, p)
    ev = np.asarray(det_evts)
    out = np.zeros((ev.shape[0], n_t, n_pos), dtype=np.float32)
    for t in range(n_t):
        cols = index_map[t]
        present = cols >= 0
        out[:, t, present] = ev[:, cols[present]]
    return out


def input_width(d, rounds, inputs):
    """Per-shot feature width for a given input choice. r*(d^2-1) per array."""
    per_array = rounds * (d ** 2 - 1)
    if inputs == 'evts':
        return per_array
    if inputs == 'evts+bits':
        return 2 * per_array
    raise ValueError(f"unknown inputs={inputs!r} (expected 'evts' or 'evts+bits')")


def assemble_features(det_bits, det_evts, inputs, student='mlp', d=5, rounds=3, p=0.010):
    """Build the student's input array from the pool arrays, as float32.

    Kept beside the model definitions so layout and architecture cannot drift apart.
    float32 avoids an implicit cast of the int8 pool arrays every batch.

    student='gru' returns [N, n_timesteps, n_positions]; anything else returns the flat
    [N, n_features] vector, where feature order is irrelevant to a Dense stack.
    """
    if student == 'gru':
        if inputs != 'evts':
            raise ValueError("the GRU student supports inputs='evts' only")
        return to_sequence(det_evts, d, rounds, p)

    if inputs == 'evts':
        x = np.asarray(det_evts)
    elif inputs == 'evts+bits':
        x = np.concatenate([np.asarray(det_evts), np.asarray(det_bits)], axis=1)
    else:
        raise ValueError(f"unknown inputs={inputs!r}")
    return x.astype(np.float32)


def _quantizers(weight_bits):
    """QKeras quantizer objects for a weight word length, or None for float layers.

    quantized_bits(B, 1, alpha=1) == ap_fixed<B,2>: 1 sign bit, 1 integer bit, B-2
    fractional bits. alpha=1 pins the scale to 1 so the fixed-point interpretation is the
    literal one hls4ml will synthesize -- QKeras's default per-channel alpha would fold a
    learned float scale in front of the weights, which is not what the FPGA implements.
    Integer split is pinned a priori here exactly as in the teacher sweep; Phase 4 range
    profiling revisits it with measured tensor ranges.
    """
    if weight_bits is None or weight_bits >= 32:
        return None
    from qkeras import quantized_bits
    return quantized_bits(weight_bits, 1, alpha=1)


def _dense(units, name, weight_bits):
    """A Dense layer, quantized if a weight word length was given."""
    q = _quantizers(weight_bits)
    if q is None:
        return KL.Dense(units, name=name)
    from qkeras import QDense
    return QDense(units, kernel_quantizer=q, bias_quantizer=q, name=name)


def _relu(name, act_bits):
    """A ReLU activation, quantized if an activation word length was given.

    quantized_relu(B, I) is unsigned (ReLU output is non-negative, so no sign bit is
    spent) with I integer bits. I is pinned to 2 here as a starting point; Phase 4
    profiling replaces it with the measured per-tensor value.
    """
    if act_bits is None or act_bits >= 32:
        return KL.Activation('relu', name=name)
    from qkeras import QActivation
    return QActivation(f'quantized_relu({act_bits},2)', name=name)


def build_mlp_student(d=5, rounds=3, inputs='evts', hidden=(128, 128),
                      weight_bits=None, act_bits=None, name='mlp_student'):
    """Flatten-and-stack student, emitting one logit per shot."""
    n_feat = input_width(d, rounds, inputs)
    x_in = KL.Input(shape=(n_feat,), name='syndrome')
    h = x_in
    for i, units in enumerate(hidden):
        h = _dense(units, f'dense_{i}', weight_bits)(h)
        h = _relu(f'relu_{i}', act_bits)(h)
    # Linear head: the output is a logit. See module docstring for why no sigmoid.
    out = _dense(1, 'logit', weight_bits)(h)
    return Model(x_in, out, name=name)


def build_gru_student(d=5, rounds=3, inputs='evts', units=64, hidden=(), p=0.010,
                      weight_bits=None, act_bits=None, name='gru_student', reset_after=False,
                      unroll=False):
    """Recurrent student: one GRU timestep per detector timestep, then a logit.

    Input is [n_timesteps, n_positions] as produced by to_sequence(), with r + 1 timesteps
    (detector_sequence_layout reads the layout off the stim circuit for d, rounds, p):
    d=5 r=5 (6, 24), d=7 r=7 (8, 48), d=9 r=9 (10, 80); d=5 r=3 is (4, 24). The scatter
    happens in the data pipeline, so this model takes the sequence directly and contains
    no reshape or gather.

    Layers: GRU(units, return_sequences=False) -> for each width in `hidden`, Dense + ReLU
    -> Dense(1) logit. `hidden` defaults to () (no Dense layers); build_student and the
    scripts' --hidden use the same default for the GRU. reset_after and unroll are passed
    to the GRU (defaults False; see below). weight_bits / act_bits swap in QGRU and QDense.

    Parameters: GRU 3 * (units * (n_positions + units) + units) with reset_after=False
    (one more bias row per gate with True), plus the Dense layers and units + 1 for the head.

    units may also be a sequence, e.g. (280, 280): stacked GRU layers, each but the last
    returning its full sequence to the next. The first layer is named 'gru' as in the
    single-layer model, the others 'gru_1', 'gru_2', ... A single int (or a one-element
    sequence) builds exactly the single-layer model.

    inputs='evts+bits' is not supported here: det_bits holds `rounds` rounds of raw
    stabilizer measurements, a different time base from det_evts' r + 1 detector
    timesteps, and there is no correct way to align them without a decision about which to
    resample.
    """
    if inputs != 'evts':
        raise ValueError("the GRU student supports inputs='evts' only "
                         f"(got {inputs!r}); det_bits has a different time base")
    n_t, n_pos, _ = detector_sequence_layout(d, rounds, p)
    seq = KL.Input(shape=(n_t, n_pos), name='syndrome_sequence')
    x_in = seq

    # reset_after passed explicitly to both layers: Keras GRU defaults to True, QKeras QGRU
    # to False, so leaving it implicit makes the float and quantized students structurally
    # different layers (differing bias shape) and their results incomparable. Default False
    # keeps every existing checkpoint loadable; True is the cuDNN-compatible form.
    # unroll=True unrolls the r+1 timesteps into plain ops, which also keeps Keras off the
    # cuDNN GRU kernel: TF 2.15's cuDNN GRU call fails under the image's cuDNN 9. Same
    # equations either way; default False.
    widths = [int(u) for u in units] if isinstance(units, (list, tuple)) else [int(units)]
    h = seq
    for i, u in enumerate(widths):
        h = _gru_layer(u, i < len(widths) - 1, 'gru' if i == 0 else f'gru_{i}', weight_bits,
                       act_bits, reset_after, unroll)(h)

    for i, n in enumerate(hidden):
        h = _dense(n, f'dense_{i}', weight_bits)(h)
        h = _relu(f'relu_{i}', act_bits)(h)
    out = _dense(1, 'logit', weight_bits)(h)
    return Model(x_in, out, name=name)


def _gru_layer(units, return_sequences, name, weight_bits, act_bits, reset_after, unroll):
    """One GRU layer, quantized (QGRU) when a weight word length is given."""
    if weight_bits is None or weight_bits >= 32:
        return KL.GRU(units, return_sequences=return_sequences, reset_after=reset_after,
                      unroll=unroll, name=name)
    else:
        from qkeras.qrecurrent import QGRU
        q = _quantizers(weight_bits)
        # State/activation quantizers are given the activation word length when one was
        # requested; at Phase 2 (act_bits=None) the recurrent activations stay float and
        # only the weights are quantized, matching the teacher sweep's weights-only stage.
        a_tanh = f'quantized_tanh({act_bits})' if act_bits else 'tanh'
        a_sigm = f'quantized_sigmoid({act_bits})' if act_bits else 'sigmoid'
        state_q = _quantizers(act_bits) if act_bits else _quantizers(weight_bits)
        return QGRU(units, return_sequences=return_sequences, reset_after=reset_after,
                    unroll=unroll, kernel_quantizer=q, recurrent_quantizer=q, bias_quantizer=q,
                    state_quantizer=state_q,
                    activation=a_tanh, recurrent_activation=a_sigm,
                    name=name)


def build_student(student, d=5, rounds=3, inputs='evts', hidden=None, units=64,
                  weight_bits=None, act_bits=None, reset_after=False, unroll=False):
    """Dispatch on the student name so trainers take a plain string argument.

    hidden=None means the architecture's default: (128, 128) for the MLP, () for the GRU.
    A given tuple, including (), is used as it is.
    """
    if student == 'mlp':
        return build_mlp_student(d=d, rounds=rounds, inputs=inputs,
                                 hidden=(128, 128) if hidden is None else tuple(hidden),
                                 weight_bits=weight_bits, act_bits=act_bits)
    if student == 'gru':
        return build_gru_student(d=d, rounds=rounds, inputs=inputs, units=units,
                                 hidden=() if hidden is None else tuple(hidden),
                                 weight_bits=weight_bits, act_bits=act_bits,
                                 reset_after=reset_after, unroll=unroll)
    raise ValueError(f"unknown student={student!r} (expected 'mlp' or 'gru')")


def logits_to_prob(logits):
    """sigmoid, for when a calibrated probability is wanted instead of a decision."""
    return tf.sigmoid(logits).numpy() if tf.is_tensor(logits) else 1.0 / (
        1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))


def student_pred_and_correct(logits, truth):
    """Decision convention for the student, mirroring eval_on_tail.rcnn_pred_and_correct.

    The student emits a logit, so the threshold is 0, which is exactly equivalent to
    thresholding sigmoid(logit) at 0.5. Single source of this convention so no downstream
    script reimplements it with a different threshold.
    """
    pred = (np.asarray(logits).reshape(-1) > 0.0).astype(np.int8)
    truth = np.asarray(truth).reshape(-1).astype(np.int8)
    return pred, (pred == truth)
