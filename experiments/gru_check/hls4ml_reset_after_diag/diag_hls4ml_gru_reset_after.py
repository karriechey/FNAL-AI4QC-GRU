# Created: 2026-09-26
# Last modified: 2026-09-26
"""Diagnostic (temporary, not research code): which Keras GRU reset_after mode does the
installed hls4ml reproduce numerically, and why?

    python diag_hls4ml_gru_reset_after.py --out <dir> [--real]

Part 1, controlled test. Two otherwise identical models, input (6, 24) like the d=5
decoder, GRU(units=4) -> Dense(1):
    A: reset_after=False        B: reset_after=True
each built twice: with Keras' default initialisation (all GRU biases zero) and with every
GRU bias set to fixed random non-zero values. For each: model.get_config(), GRU weight
names/shapes, TF inference, hls4ml conversion (the same path as the C2 check: Vivado
backend, io_parallel, granularity 'name', every precision ap_fixed<32,16>, GRU tables
65,536 entries of ap_fixed<32,16>), compile, hls4ml inference on 8 fixed random 0/1
inputs, max |TF - HLS|. Any failure keeps its full traceback.
For the reset_after=False models it also evaluates, in numpy, the reset_after=True
equations with the same kernel / recurrent kernel and the same (3u,) bias used as both
b and br. If hls4ml tracks that instead of Keras, the backend is running the
reset_after=True equations.
The generated firmware bias arrays (b<N>.h, br<N>.h) are read to report their lengths.

Part 1b, the equation error alone (numpy): each reset_after=False model's Keras output vs
the reset_after=True equations on the same weights with the bias used correctly - what the
Vivado kernel would compute if only the converter's bias split were fixed.

Part 2 (--real): the same conversion of real trained checkpoints on 500 validation-block
shots: the d=5 u140 reset_after=False run of record, the 20M-augmented reset_after=False
seed 0, and the reset_after=True (unroll=True) d=5 seed 0 run. Nothing is retrained.
"""
import argparse
import glob
import json
import os
import re
import sys
import traceback

import numpy as np

REPO = os.path.expanduser('~/FNAL-AI4QC-GRU')
sys.path.insert(0, os.path.join(REPO, 'src'))
PREC = 'ap_fixed<32,16>'
T, F = 6, 24


def log(*a):
    print(*a, flush=True)


def hls_config(hls4ml, model):
    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name',
                                               default_precision=PREC, backend='Vivado')
    gru = [k for k in cfg['LayerName'] if 'gru' in k.lower()][0]
    cfg['LayerName'][gru]['TableSize'] = 65536
    cfg['LayerName'][gru]['Precision']['table'] = PREC
    return cfg


def firmware_bias_lengths(outdir):
    res = {}
    for f in sorted(glob.glob(os.path.join(outdir, 'firmware', 'weights', 'b*.h'))):
        txt = open(f).read()
        m = re.search(r'\[(\d*)\]', txt)
        n_vals = len(re.findall(r'-?\d+\.\d+(?:e-?\d+)?', txt.split('{', 1)[1])) if '{' in txt else None
        res[os.path.basename(f)] = dict(declared=m.group(1) if m else None, values=n_vals)
    return res


def np_gru(x, W, U, b, br, reset_after, u):
    """Keras GRU equations (gate order z, r, h) with bias b (input) and br (recurrent)."""
    sig = lambda v: 1 / (1 + np.exp(-v))
    h = np.zeros((x.shape[0], u))
    for t in range(x.shape[1]):
        xw = x[:, t] @ W + b
        if reset_after:
            hu = h @ U + br
            z = sig(xw[:, :u] + hu[:, :u]); r = sig(xw[:, u:2 * u] + hu[:, u:2 * u])
            c = np.tanh(xw[:, 2 * u:] + r * hu[:, 2 * u:])
        else:
            hu = h @ U[:, :2 * u]
            z = sig(xw[:, :u] + hu[:, :u]); r = sig(xw[:, u:2 * u] + hu[:, u:])
            c = np.tanh(xw[:, 2 * u:] + (r * h) @ U[:, 2 * u:])
        h = z * h + (1 - z) * c
    return h


def convert_and_compare(tf, hls4ml, model, x, outdir, rec):
    try:
        cfg = hls_config(hls4ml, model)
        hm = hls4ml.converters.convert_from_keras_model(model, hls_config=cfg, output_dir=outdir,
                                                        backend='Vivado', io_type='io_parallel')
        rec['hls_conversion'] = 'PASS'
        gl = [l for l in hm.get_layers() if l.class_name == 'GRU'][0]
        rec['ir_apply_reset_gate'] = gl.get_attr('apply_reset_gate')
        rec['ir_bias_shape'] = list(np.shape(gl.get_weights('bias').data))
        rec['ir_recurrent_bias_shape'] = list(np.shape(gl.get_weights('recurrent_bias').data))
    except Exception:
        rec['hls_conversion'] = 'FAIL'; rec['traceback'] = traceback.format_exc(); return None
    try:
        hm.compile(); rec['hls_compile'] = 'PASS'
    except Exception:
        rec['hls_compile'] = 'FAIL'; rec['traceback'] = traceback.format_exc(); return None
    rec['firmware_bias_arrays'] = firmware_bias_lengths(outdir)
    try:
        y = np.asarray(hm.predict(np.ascontiguousarray(x, dtype=np.float32))).reshape(-1)
        rec['hls_prediction'] = 'PASS'
        return y
    except Exception:
        rec['hls_prediction'] = 'FAIL'; rec['traceback'] = traceback.format_exc(); return None


def part1(tf, hls4ml, out):
    KL = tf.keras.layers
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=(8, T, F)).astype(np.float32)
    results = {}
    for bias_mode in ('keras_default_zero_bias', 'random_nonzero_bias'):
        for ra in (False, True):
            name = f'units4_reset_after_{ra}_{bias_mode}'
            log(f'\n===== {name}')
            tf.keras.utils.set_random_seed(123)
            inp = KL.Input(shape=(T, F))
            h = KL.GRU(4, reset_after=ra, activation='tanh', recurrent_activation='sigmoid',
                       name='gru')(inp)
            m = tf.keras.Model(inp, KL.Dense(1, name='logit')(h))
            if bias_mode == 'random_nonzero_bias':
                W, U, b = m.get_layer('gru').get_weights()
                b = np.random.default_rng(7).normal(0, 0.5, size=b.shape).astype(np.float32)
                m.get_layer('gru').set_weights([W, U, b])
            rec = dict(name=name, reset_after=ra, bias_mode=bias_mode)
            gcfg = m.get_layer('gru').get_config()
            rec['gru_config'] = {k: gcfg[k] for k in ('units', 'activation', 'recurrent_activation',
                                                      'reset_after', 'use_bias', 'unroll')}
            rec['gru_weights'] = {w.name: list(w.shape) for w in m.get_layer('gru').weights}
            log('model.get_config() GRU:', json.dumps(rec['gru_config']))
            log('GRU weights:', rec['gru_weights'])
            y_tf = m.predict(x, verbose=0).reshape(-1)
            rec['keras_inference'] = 'PASS'
            y_hls = convert_and_compare(tf, hls4ml, m, x, os.path.join(out, 'hls_' + name), rec)
            W, U, b = [np.asarray(v, np.float64) for v in m.get_layer('gru').get_weights()]
            Wd, bd = [np.asarray(v, np.float64) for v in m.get_layer('logit').get_weights()]
            if ra:
                ref_np = np_gru(x, W, U, b[0], b[1], True, 4) @ Wd + bd
            else:
                ref_np = np_gru(x, W, U, b, None, False, 4) @ Wd + bd
                # what a reset_after=True kernel would compute with these weights
                alt = np_gru(x, W, U, b, b, True, 4) @ Wd + bd
                rec['numpy_reset_after_True_equations_on_these_weights'] = alt.reshape(-1).tolist()
            rec['numpy_reference_matches_keras_max_diff'] = float(np.max(np.abs(ref_np.reshape(-1) - y_tf)))
            rec['keras_logits'] = y_tf.tolist()
            if y_hls is not None:
                rec['hls_logits'] = y_hls.tolist()
                rec['max_abs_tf_minus_hls'] = float(np.max(np.abs(y_tf - y_hls)))
                if not ra:
                    rec['max_abs_hls_minus_numpy_reset_after_True'] = float(
                        np.max(np.abs(y_hls - np.asarray(rec['numpy_reset_after_True_equations_on_these_weights']))))
            for k in ('keras_inference', 'hls_conversion', 'hls_compile', 'hls_prediction',
                      'max_abs_tf_minus_hls', 'ir_apply_reset_gate', 'ir_bias_shape',
                      'ir_recurrent_bias_shape', 'firmware_bias_arrays',
                      'max_abs_hls_minus_numpy_reset_after_True'):
                if k in rec:
                    log(f'  {k}: {rec[k]}')
            if 'traceback' in rec:
                log(rec['traceback'])
            results[name] = rec
    return results


def part1b(tf, hls4ml, out):
    """Equation error alone, without the bias bug (numpy; hls4ml not involved).

    For each reset_after=False model: Keras output vs the reset_after=True equations run on
    the SAME weights with the bias used correctly (input bias = the (3u,) bias, recurrent
    bias = 0). This is what the Vivado GRU kernel would compute if only the converter's
    bias handling were fixed. (Repairing the bias inside hls4ml's in-memory model is not
    possible without backend-specific attributes: the Vivado writer then fails with
    "'WeightVariable' object has no attribute 'storage'".)
    """
    KL = tf.keras.layers
    x4 = np.random.default_rng(0).integers(0, 2, size=(8, T, F)).astype(np.float32)
    cases = []
    for bias_mode in ('keras_default_zero_bias', 'random_nonzero_bias'):
        tf.keras.utils.set_random_seed(123)
        inp = KL.Input(shape=(T, F))
        m = tf.keras.Model(inp, KL.Dense(1, name='logit')(KL.GRU(4, reset_after=False, name='gru')(inp)))
        if bias_mode == 'random_nonzero_bias':
            W, U, b = m.get_layer('gru').get_weights()
            m.get_layer('gru').set_weights([W, U, np.random.default_rng(7).normal(0, 0.5, size=b.shape).astype(np.float32)])
        cases.append((f'units4_{bias_mode}', m, x4))
    import partitions as P
    from StudentModels import build_gru_student, to_sequence
    va = P.block('val')
    xr = to_sequence(np.load(P.pool_path(5, 'formal'))['det_evts'][va[0]:va[0] + 500], 5, 5, 0.004).astype(np.float32)
    R = os.path.expanduser('~/rcnn_threshold/results')
    for name, w in (('real_d5_u140_10M_run_of_record', f'{R}/gh200_gru_d5_e200_20260819T185011Z/seed0/ckpt/gru_d5_p004_u140_hard_seed0_ntr10000000_b10000.best.weights.h5'),
                    ('real_d5_u140_20M_aug_seed0', f'{R}/d5_u140_ntr20M_aug_20260821T154820Z/seed0/ckpt/gru_d5_p004_u140_ext5000000_hard_seed0_ntr15000000_b10000.best.weights.h5')):
        m = build_gru_student(d=5, rounds=5, units=140, p=0.004)
        m.load_weights(w)
        cases.append((name, m, xr))
    results = {}
    for name, m, x in cases:
        W, U, b = [np.asarray(v, np.float64) for v in m.get_layer('gru').get_weights()]
        Wd, bd = [np.asarray(v, np.float64) for v in m.get_layer('logit').get_weights()]
        u = U.shape[0]
        y_tf = m.predict(x, verbose=0).reshape(-1)
        alt = (np_gru(x, W, U, b, np.zeros_like(b), True, u) @ Wd + bd).reshape(-1)
        rec = dict(max_abs_keras_False_minus_reset_after_True_equations=float(np.max(np.abs(y_tf - alt))),
                   mean_abs=float(np.mean(np.abs(y_tf - alt))),
                   sign_agreement=float(np.mean(np.sign(y_tf) == np.sign(alt))),
                   keras_logit_std=float(np.std(y_tf)), n=int(len(y_tf)))
        log(f'\n===== equation-only error, {name}: ' + json.dumps(rec))
        results[name] = rec
    return results


def part2(tf, hls4ml, out):
    import partitions as P
    from StudentModels import build_gru_student, to_sequence
    va = P.block('val')
    det = np.load(P.pool_path(5, 'formal'))['det_evts'][va[0]:va[0] + 500]
    x = to_sequence(det, 5, 5, 0.004).astype(np.float32)
    R = os.path.expanduser('~/rcnn_threshold/results')
    Q = os.path.join(REPO, 'experiments', 'queue_ra1_20M')
    cases = {
        'd5_u140_10M_run_of_record_reset_after_False': (
            f'{R}/gh200_gru_d5_e200_20260819T185011Z/seed0/ckpt/gru_d5_p004_u140_hard_seed0_ntr10000000_b10000.best.weights.h5', False, False),
        'd5_u140_20M_aug_seed0_reset_after_False': (
            f'{R}/d5_u140_ntr20M_aug_20260821T154820Z/seed0/ckpt/gru_d5_p004_u140_ext5000000_hard_seed0_ntr15000000_b10000.best.weights.h5', False, False),
        'd5_u140_20M_aug_seed0_reset_after_True_unroll': (
            glob.glob(f'{Q}/d5_s0_20260926T174023Z/seed0/ckpt/*.best.weights.h5')[0], True, True),
    }
    results = {}
    for name, (w, ra, ur) in cases.items():
        log(f'\n===== REAL {name}')
        rec = dict(name=name, weights=w)
        m = build_gru_student(d=5, rounds=5, units=140, p=0.004, reset_after=ra, unroll=ur)
        m.load_weights(w)
        g = m.get_layer('gru')
        rec['gru_config'] = {k: g.get_config()[k] for k in ('units', 'reset_after', 'unroll', 'activation', 'recurrent_activation')}
        rec['gru_weights'] = {v.name: list(v.shape) for v in g.weights}
        log('GRU config:', rec['gru_config']); log('GRU weights:', rec['gru_weights'])
        y_tf = m.predict(x, verbose=0).reshape(-1)
        rec['keras_inference'] = 'PASS'
        y = convert_and_compare(tf, hls4ml, m, x, os.path.join(out, 'hls_real_' + name), rec)
        if y is not None:
            rec['max_abs_tf_minus_hls'] = float(np.max(np.abs(y_tf - y)))
            rec['mean_abs_tf_minus_hls'] = float(np.mean(np.abs(y_tf - y)))
            rec['sign_agreement'] = float(np.mean(np.sign(y_tf) == np.sign(y)))
            rec['keras_logit_std'] = float(np.std(y_tf))
        for k in ('hls_conversion', 'hls_compile', 'hls_prediction', 'ir_apply_reset_gate',
                  'ir_bias_shape', 'ir_recurrent_bias_shape', 'firmware_bias_arrays',
                  'max_abs_tf_minus_hls', 'mean_abs_tf_minus_hls', 'sign_agreement', 'keras_logit_std'):
            if k in rec:
                log(f'  {k}: {rec[k]}')
        if 'traceback' in rec:
            log(rec['traceback'])
        results[name] = rec
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--real', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf
    tf.config.experimental.enable_tensor_float_32_execution(False)
    import hls4ml
    env = dict(python=sys.version.split()[0], tensorflow=tf.__version__,
               tf_keras_module=getattr(tf.keras, '__name__', str(tf.keras)),
               hls4ml=hls4ml.__version__, hls4ml_path=os.path.dirname(hls4ml.__file__),
               TF_USE_LEGACY_KERAS=os.environ.get('TF_USE_LEGACY_KERAS'),
               devices=[d.name for d in tf.config.list_physical_devices()])
    log('ENV', json.dumps(env))
    rep = dict(env=env, part1=part1(tf, hls4ml, a.out), part1b=part1b(tf, hls4ml, a.out))
    if a.real:
        rep['part2'] = part2(tf, hls4ml, a.out)
    with open(os.path.join(a.out, 'report.json'), 'w') as fh:
        json.dump(rep, fh, indent=1, default=str)
    log(f"\nwrote {os.path.join(a.out, 'report.json')}")


if __name__ == '__main__':
    main()
