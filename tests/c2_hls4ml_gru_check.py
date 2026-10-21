# Created: 2026-09-26
# Last modified: 2026-09-26

"""C2: does hls4ml's GRU match Keras numerically, for reset_after True and False?

u140 and u70 at d=5 with seeded random weights; Vivado, io_parallel, ap_fixed<32,16>; C-sim on
the first 500 d=5 eval-block shots. Reports max |logit_hls - logit_keras| and sign agreement.
--table-size and --table-precision separate table error from wrong equations; --trained adds
the d=5 u140 checkpoint.

~/tfenv/bin/python tests/c2_hls4ml_gru_check.py --out experiments/gru_check/c2_<stamp>
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

D, N_SHOTS, PRECISION = 5, 500, 'ap_fixed<32,16>'


def build(tf, units, reset_after, seed, unroll=False):
    tf.keras.utils.set_random_seed(seed)
    KL = tf.keras.layers
    x = KL.Input(shape=(D + 1, D * D - 1), name='syndrome_sequence')
    h = KL.GRU(units, return_sequences=False, reset_after=reset_after, unroll=unroll,
               name='gru')(x)
    y = KL.Dense(1, name='logit')(h)
    return tf.keras.Model(x, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--units', type=int, nargs='*', default=[140, 70])
    ap.add_argument('--table-size', type=int, default=None, help='GRU activation table size')
    ap.add_argument('--table-precision', default=None, help='GRU activation table type')
    ap.add_argument('--trained', action='store_true',
                    help='also check the d=5 u140 run-of-record best checkpoint')
    ap.add_argument('--trained-weights', default=None,
                    help='check this d=5 .weights.h5 instead of the random-weight models')
    ap.add_argument('--trained-units', type=int, default=140)
    ap.add_argument('--trained-reset-after', action='store_true')
    ap.add_argument('--trained-unroll', action='store_true')
    ap.add_argument('--no-tf32', action='store_true',
                    help='Keras reference logits with TF32 off (no effect on a CPU-only run)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf
    if a.no_tf32:
        tf.config.experimental.enable_tensor_float_32_execution(False)
    import hls4ml
    import partitions as P
    from StudentModels import to_sequence

    e0 = P.block('eval')[0]
    det = np.load(P.pool_path(D, 'formal'))['det_evts'][e0:e0 + N_SHOTS]
    x = np.ascontiguousarray(to_sequence(det, D, D, P.load()['p']), dtype=np.float32)
    results = dict(hls4ml=hls4ml.__version__, tensorflow=tf.__version__, precision=PRECISION,
                   table_size=a.table_size or 'default 1024',
                   table_precision=a.table_precision or 'default fixed<18,8>',
                   inputs=f'first {N_SHOTS} shots of the d={D} eval block [{e0}, ...)',
                   keras_reference=('CPU' if not tf.config.list_physical_devices('GPU') else 'GPU')
                   + (', TF32 off' if a.no_tf32 else ''),
                   models={})

    cases = [(u, ra, None) for u in a.units for ra in (True, False)]
    if a.trained_weights:
        cases = [(a.trained_units, a.trained_reset_after, a.trained_weights)]
    if a.trained:
        paths = P._load('paths.yaml')
        ck = os.path.join(os.path.expanduser(paths['results_root']),
                          paths['runs_of_record']['d5_u140_10M'], 'seed0', 'ckpt',
                          'gru_d5_p004_u140_hard_seed0_ntr10000000_b10000.best.weights.h5')
        cases.append((140, False, ck))
    for units, reset_after, ckpt in cases:
            name = f'u{units}_reset_after_{reset_after}' + (
                ('_trained_' + os.path.basename(ckpt).split('.')[0]) if ckpt else '')
            rec = dict(units=units, reset_after=reset_after, weights=ckpt or f'random, seed {units}')
            try:
                model = build(tf, units, reset_after, seed=units,
                              unroll=bool(ckpt) and a.trained_unroll)
                if ckpt:
                    model.load_weights(ckpt)
                rec['params'] = int(model.count_params())
                rec['bias_shape'] = list(model.get_layer('gru').cell.bias.shape)
                y_keras = model.predict(x, batch_size=N_SHOTS, verbose=0).reshape(-1)
                cfg = hls4ml.utils.config_from_keras_model(
                    model, granularity='name', default_precision=PRECISION, backend='Vivado')
                if a.table_size:
                    cfg['LayerName']['gru']['TableSize'] = a.table_size
                if a.table_precision:
                    cfg['LayerName']['gru']['Precision']['table'] = a.table_precision
                hm = hls4ml.converters.convert_from_keras_model(
                    model, hls_config=cfg, output_dir=os.path.join(a.out, f'hls_{name}'),
                    backend='Vivado', io_type='io_parallel')
                t0 = time.time()
                hm.compile()
                rec['compile_s'] = round(time.time() - t0, 1)
                y_hls = np.asarray(hm.predict(x)).reshape(-1)
                diff = np.abs(y_hls - y_keras)
                rec.update(max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()),
                           sign_agreement=float(np.mean(np.sign(y_hls) == np.sign(y_keras))),
                           keras_logit_std=float(y_keras.std()))
                np.savez(os.path.join(a.out, f'{name}.npz'), keras=y_keras, hls=y_hls)
            except Exception as exc:
                rec['error'] = f'{type(exc).__name__}: {exc}'
                rec['traceback'] = traceback.format_exc()[-3000:]
            results['models'][name] = rec
            print(f"[C2] {name}: " + ', '.join(f'{k}={v}' for k, v in rec.items()
                                                if k != 'traceback'), flush=True)
            with open(os.path.join(a.out, 'c2_results.json'), 'w') as fh:
                json.dump(results, fh, indent=1)
    print(f"[C2] wrote {os.path.join(a.out, 'c2_results.json')}")


if __name__ == '__main__':
    main()
