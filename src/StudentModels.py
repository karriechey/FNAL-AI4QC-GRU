#!/usr/bin/env python3
# from FNAL-QCDecoding-FPGA/StudentModels.py @ d5ee86f
# Created: 2026-07-29
# Last modified: 2026-10-29 (layout cache keyed by basis). Earlier 2026-10-05 (units may be a sequence: stacked GRU layers; a single int builds exactly the model as before).
# Earlier 2026-09-26 (reset_after and unroll are build arguments, default False)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras import Model


def _patch_qkeras_recurrent_nest():
    try:
        from tensorflow.python.util import nest as _nest
    except ImportError:  # pragma: no cover - TF internals moved; let QKeras fail loudly
        return
    if not hasattr(_nest, 'is_sequence') and hasattr(_nest, 'is_nested'):
        _nest.is_sequence = _nest.is_nested


_patch_qkeras_recurrent_nest()


_LAYOUT_CACHE = {}


def detector_sequence_layout(d, rounds, p):
    # Map flat det_evts onto (timestep, plaquette position), zero where unmeasured, so the
    # input stays rectangular for hls4ml. Returns (n_timesteps, n_positions, index_map).

    from circuit_generators import current_basis
    key = (d, rounds, round(p, 6), current_basis())
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
    # Scatter flat det_evts [N, n_det] into [N, n_timesteps, n_positions], zero-filled.
    n_t, n_pos, index_map = detector_sequence_layout(d, rounds, p)
    ev = np.asarray(det_evts)
    out = np.zeros((ev.shape[0], n_t, n_pos), dtype=np.float32)
    for t in range(n_t):
        cols = index_map[t]
        present = cols >= 0
        out[:, t, present] = ev[:, cols[present]]
    return out


def input_width(d, rounds, inputs):
    # Per-shot feature width for a given input choice. r*(d^2-1) per array.
    per_array = rounds * (d ** 2 - 1)
    if inputs == 'evts':
        return per_array
    if inputs == 'evts+bits':
        return 2 * per_array
    raise ValueError(f"unknown inputs={inputs!r} (expected 'evts' or 'evts+bits')")


def assemble_features(det_bits, det_evts, inputs, student='mlp', d=5, rounds=3, p=0.010):
    # student input array from the pool arrays, float32

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
    # QKeras quantizers for a weight word length, or None for float layers.
    # quantized_bits(B, 1, alpha=1) == ap_fixed<B,2>: sign bit, 1 integer bit, B-2 fractional
    
    if weight_bits is None or weight_bits >= 32:
        return None
    from qkeras import quantized_bits
    return quantized_bits(weight_bits, 1, alpha=1)


def _dense(units, name, weight_bits):
    q = _quantizers(weight_bits)
    if q is None:
        return KL.Dense(units, name=name)
    from qkeras import QDense
    return QDense(units, kernel_quantizer=q, bias_quantizer=q, name=name)


def _relu(name, act_bits):
    # quantized_relu(B, 2): unsigned, 2 integer bits
    if act_bits is None or act_bits >= 32:
        return KL.Activation('relu', name=name)
    from qkeras import QActivation
    return QActivation(f'quantized_relu({act_bits},2)', name=name)


def build_mlp_student(d=5, rounds=3, inputs='evts', hidden=(128, 128),
                      weight_bits=None, act_bits=None, name='mlp_student'):
    n_feat = input_width(d, rounds, inputs)
    x_in = KL.Input(shape=(n_feat,), name='syndrome')
    h = x_in
    for i, units in enumerate(hidden):
        h = _dense(units, f'dense_{i}', weight_bits)(h)
        h = _relu(f'relu_{i}', act_bits)(h)
    # linear head: the output is a logit
    out = _dense(1, 'logit', weight_bits)(h)
    return Model(x_in, out, name=name)


def build_gru_student(d=5, rounds=3, inputs='evts', units=64, hidden=(), p=0.010,
                      weight_bits=None, act_bits=None, name='gru_student', reset_after=False,
                      unroll=False):
    # GRU over r + 1 detector timesteps, input (n_timesteps, n_positions) from to_sequence:
    # d5 r5 (6, 24), d7 r7 (8, 48), d9 r9 (10, 80). GRU -> Dense + ReLU per `hidden` -> logit.
    # units may be a sequence for stacked GRU layers ('gru', 'gru_1', ...).
    # GRU params (reset_after=False): 3 * (units * (n_positions + units) + units).
    # weight_bits / act_bits switch to QGRU and QDense. inputs='evts+bits' is not supported.
    if inputs != 'evts':
        raise ValueError("the GRU student supports inputs='evts' only "
                         f"(got {inputs!r}); det_bits has a different time base")
    n_t, n_pos, _ = detector_sequence_layout(d, rounds, p)
    seq = KL.Input(shape=(n_t, n_pos), name='syndrome_sequence')
    x_in = seq

    # reset_after is set explicitly: Keras GRU defaults to True, QKeras QGRU to False.
    # False keeps old checkpoints loadable; True is the cuDNN form.
    # unroll=True keeps Keras off the cuDNN GRU kernel, which fails under the image's cuDNN 9.
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
    if weight_bits is None or weight_bits >= 32:
        return KL.GRU(units, return_sequences=return_sequences, reset_after=reset_after,
                      unroll=unroll, name=name)
    else:
        from qkeras.qrecurrent import QGRU
        q = _quantizers(weight_bits)
        # act_bits=None: recurrent activations stay float, weights-only quantization
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
    # hidden=None -> architecture default: (128, 128) for MLP, () for GRU
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
    return tf.sigmoid(logits).numpy() if tf.is_tensor(logits) else 1.0 / (
        1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))


def student_pred_and_correct(logits, truth):
    # decision rule: logit > 0 (== sigmoid > 0.5); the one place it is defined
    pred = (np.asarray(logits).reshape(-1) > 0.0).astype(np.int8)
    truth = np.asarray(truth).reshape(-1).astype(np.int8)
    return pred, (pred == truth)
