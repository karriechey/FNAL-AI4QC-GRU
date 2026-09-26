# Created: 2026-09-26
# Last modified: 2026-09-26
"""C2: does hls4ml's GRU reproduce Keras numerically, for reset_after=True and False?

Earlier (hls4ml 1.3.0, Mac) a reset_after=False GRU converted without error, but the
converter split the (3u,) bias as if it were the (2, 3u) reset_after=True bias, and the
Vivado templates compute the reset_after=True update. C-sim could not be compiled on the
Mac, so the effect on outputs was never measured. This measures it.

For units 140 and 70, reset_after True and False (4 models, d=5, input (6, 24)):
  - build GRU(units) -> Dense(1) with seeded random weights (FP32)
  - convert with hls4ml, Vivado backend, io_parallel, every precision ap_fixed<32,16>
  - hls_model.compile() (C simulation, g++)
  - predict on 500 real inputs: the first 500 shots of the d=5 eval block, scattered to
    (6, 24) with the repo's to_sequence
  - report max |logit_hls - logit_keras| and the fraction of shots where sign(logit) agrees

The activation look-up tables still limit accuracy for the reset_after=True model, so its
difference sets the floor that a correct conversion reaches at this precision.

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


def build(tf, units, reset_after, seed):
    tf.keras.utils.set_random_seed(seed)
    KL = tf.keras.layers
    x = KL.Input(shape=(D + 1, D * D - 1), name='syndrome_sequence')
    h = KL.GRU(units, return_sequences=False, reset_after=reset_after, name='gru')(x)
    y = KL.Dense(1, name='logit')(h)
    return tf.keras.Model(x, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--units', type=int, nargs='*', default=[140, 70])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf
    import hls4ml
    import partitions as P
    from StudentModels import to_sequence

    e0 = P.block('eval')[0]
    det = np.load(P.pool_path(D, 'formal'))['det_evts'][e0:e0 + N_SHOTS]
    x = np.ascontiguousarray(to_sequence(det, D, D, P.load()['p']), dtype=np.float32)
    results = dict(hls4ml=hls4ml.__version__, tensorflow=tf.__version__, precision=PRECISION,
                   inputs=f'first {N_SHOTS} shots of the d={D} eval block [{e0}, ...)',
                   models={})

    for units in a.units:
        for reset_after in (True, False):
            name = f'u{units}_reset_after_{reset_after}'
            rec = dict(units=units, reset_after=reset_after)
            try:
                model = build(tf, units, reset_after, seed=units)
                rec['params'] = int(model.count_params())
                rec['bias_shape'] = list(model.get_layer('gru').cell.bias.shape)
                y_keras = model.predict(x, batch_size=N_SHOTS, verbose=0).reshape(-1)
                cfg = hls4ml.utils.config_from_keras_model(
                    model, granularity='name', default_precision=PRECISION, backend='Vivado')
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
