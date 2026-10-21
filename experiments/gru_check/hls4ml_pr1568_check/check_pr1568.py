# Created: 2026-09-27
# Last modified: 2026-09-27

"""Check hls4ml PR #1568 (Vivado GRU reset_after=False) on trained d=5 checkpoints.

Each model is converted with the default table ranges and with the sigmoid/tanh tables widened
to +-32 (edited in the generated project only), and compared with Keras CPU logits on the first
10,000 d=5 eval-block shots.

python3 check_pr1568.py --out <dir>      (qdec:tf215-hlsfix image)
"""
import argparse
import glob
import json
import os
import re
import sys
import time
import traceback

import numpy as np

REPO = os.path.expanduser('~/FNAL-AI4QC-GRU')
sys.path.insert(0, os.path.join(REPO, 'src'))
PREC = 'ap_fixed<32,16>'
N_SHOTS = 10_000


def widen_tables(project_dir, new_range=32.0):
    # widen the generated sigmoid/tanh tables to +-new_range (init range, lookup scale and offset)
    path = os.path.join(project_dir, 'firmware', 'nnet_utils', 'nnet_activation.h')
    s = open(path).read()
    R = int(new_range)
    edits = [
        ('init_sigmoid_table', 'float in_val = 2 * 8.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);',
         f'float in_val = 2 * {new_range} * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);'),
        ('void sigmoid(data_T data[CONFIG_T::n_in]', 'index = data_round + 8 * CONFIG_T::table_size / 16;',
         f'index = data_round + {R} * CONFIG_T::table_size / {2 * R};'),
        ('void sigmoid(data_T data[CONFIG_T::n_in]', 'data_round = data[ii] * CONFIG_T::table_size / 16;',
         f'data_round = data[ii] * CONFIG_T::table_size / {2 * R};'),
        ('init_tanh_table', 'float in_val = 2 * 4.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);',
         f'float in_val = 2 * {new_range} * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);'),
        ('void tanh(data_T data[CONFIG_T::n_in]', 'index = data_round + 4 * CONFIG_T::table_size / 8;',
         f'index = data_round + {R} * CONFIG_T::table_size / {2 * R};'),
        ('void tanh(data_T data[CONFIG_T::n_in]', 'data_round = data[ii] * CONFIG_T::table_size / 8;',
         f'data_round = data[ii] * CONFIG_T::table_size / {2 * R};'),
    ]
    for func, old, new in edits:
        start = s.index(func)
        end = s.find('\ntemplate', start + 1)
        seg = s[start:end]
        assert seg.count(old) == 1, (func, old, seg.count(old))
        s = s[:start] + seg.replace(old, new) + s[end:]
    open(path, 'w').write(s)


def run_case(tf, hls4ml, model, x, y_tf, outdir, wide):
    rec = {}
    try:
        cfg = hls4ml.utils.config_from_keras_model(model, granularity='name',
                                                   default_precision=PREC, backend='Vivado')
        cfg['LayerName']['gru']['TableSize'] = 65536
        cfg['LayerName']['gru']['Precision']['table'] = PREC
        hm = hls4ml.converters.convert_from_keras_model(model, hls_config=cfg, output_dir=outdir,
                                                        backend='Vivado', io_type='io_parallel')
        gl = [l for l in hm.get_layers() if l.class_name == 'GRU'][0]
        rec['ir_apply_reset_gate'] = gl.get_attr('apply_reset_gate')
        rec['ir_bias_shape'] = list(np.shape(gl.get_weights('bias').data))
        rec['ir_recurrent_bias_shape'] = list(np.shape(gl.get_weights('recurrent_bias').data))
        hm.write()
        if wide:
            widen_tables(outdir)
        t0 = time.time()
        hm._compile()
        rec['compile_s'] = round(time.time() - t0, 1)
        cfg_h = open(os.path.join(outdir, 'firmware', 'parameters.h')).read()
        m = re.search(r'static const bool reset_after = (\w+);', cfg_h)
        rec['firmware_reset_after'] = m.group(1) if m else 'not set'
        y = np.asarray(hm.predict(np.ascontiguousarray(x, dtype=np.float32))).reshape(-1)
        d = np.abs(y - y_tf)
        rec.update(max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()),
                   sign_agreement=float(np.mean(np.sign(y) == np.sign(y_tf))),
                   decisions_differ=int(np.sum((y > 0) != (y_tf > 0))), n=int(len(y)),
                   keras_logit_std=float(y_tf.std()))
    except Exception:
        rec['error'] = traceback.format_exc()[-3000:]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.experimental.enable_tensor_float_32_execution(False)
    import hls4ml
    import partitions as P
    from StudentModels import build_gru_student, to_sequence
    e0 = P.block('eval')[0]
    x = to_sequence(np.load(P.pool_path(5, 'formal'))['det_evts'][e0:e0 + N_SHOTS], 5, 5, 0.004)
    R = os.path.expanduser('~/rcnn_threshold/results')
    Q = os.path.join(REPO, 'experiments', 'queue_ra1_20M')
    models = {
        'u140_reset_after_False_10M_run_of_record': (140, False, False,
            f'{R}/gh200_gru_d5_e200_20260819T185011Z/seed0/ckpt/gru_d5_p004_u140_hard_seed0_ntr10000000_b10000.best.weights.h5'),
        'u140_reset_after_False_20M_aug_seed0': (140, False, False,
            f'{R}/d5_u140_ntr20M_aug_20260821T154820Z/seed0/ckpt/gru_d5_p004_u140_ext5000000_hard_seed0_ntr15000000_b10000.best.weights.h5'),
        'u70_reset_after_False_gkd20m_hard_seed0': (70, False, False,
            os.path.expanduser('~/reference_ckpts/gkd20m_u70_hard_seed0/gkd20m_u70_hard_seed0.best.weights.h5')),
        'u140_reset_after_True_unroll_20M_aug_seed0': (140, True, True,
            glob.glob(f'{Q}/d5_s0_20260926T174023Z/seed0/ckpt/*.best.weights.h5')[0]),
    }
    rep = dict(hls4ml=hls4ml.__version__, hls4ml_path=os.path.dirname(hls4ml.__file__),
               tensorflow=tf.__version__, shots=f'first {N_SHOTS} of the d=5 eval block [{e0}, ...)',
               keras_reference='CPU, TF32 off', results={})
    for name, (units, ra, ur, w) in models.items():
        m = build_gru_student(d=5, rounds=5, units=units, p=0.004, reset_after=ra, unroll=ur)
        m.load_weights(w)
        y_tf = m.predict(x, batch_size=10_000, verbose=0).reshape(-1)
        for variant, wide in (('default_tables', False), ('wide32_tables', True)):
            key = f'{name}__{variant}'
            rec = dict(weights=w, reset_after=ra, unroll=ur, units=units,
                       **run_case(tf, hls4ml, m, x, y_tf, os.path.join(a.out, 'hls_' + key), wide))
            rep['results'][key] = rec
            print(f"[pr1568] {key}: " + ', '.join(f'{k}={v}' for k, v in rec.items()
                                                  if k not in ('weights', 'error')), flush=True)
            if 'error' in rec:
                print(rec['error'], flush=True)
            with open(os.path.join(a.out, 'report.json'), 'w') as fh:
                json.dump(rep, fh, indent=1)


if __name__ == '__main__':
    main()
