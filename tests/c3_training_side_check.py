# Created: 2026-09-26
# Last modified: 2026-09-26
"""C3: training-side check of reset_after=True vs False on the GH200.

Run inside the qdec:tf215 image with the launcher's determinism variables:

    podman run ... --device nvidia.com/gpu=all -e TF_DETERMINISTIC_OPS=1 \
        -e TF_CUDNN_DETERMINISTIC=1 qdec:tf215 python3 tests/c3_training_side_check.py --out <dir>

Reports:
  1 parameter counts for GRU(u) -> Dense(1) with reset_after True and False
    (u140 at d=5/7/9, and u280 at d=9)
  2 at d=7, batch 10,000, u140, for each reset_after setting:
      - whether the cuDNN GRU kernel actually runs: a TensorFlow profiler trace of three
        training steps is searched for the CudnnRNN op
      - steps/s over two timed epochs of 50 steps (real d=7 training-block shots)
      - whether two identical 20-step trainings (same seed) give bit-identical weights
  3 QKeras QGRU with reset_after=True builds and runs a forward pass (8-bit weights),
    and the existing reset_after=False quantized path (build_gru_student) still runs with
    the repo's nest.is_sequence shim.
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from helpers import assert_deterministic_env  # noqa: E402

N_TIME, BATCH, UNITS = 500_000, 10_000, 140


def gru_model(tf, d, units, reset_after, seed=0):
    tf.keras.utils.set_random_seed(seed)
    KL = tf.keras.layers
    x = KL.Input(shape=(d + 1, d * d - 1))
    h = KL.GRU(units, return_sequences=False, reset_after=reset_after, name='gru')(x)
    return tf.keras.Model(x, KL.Dense(1, name='logit')(h))


def compiled(tf, model):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu-mem-mib', type=int, default=3500)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    assert_deterministic_env()
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=a.gpu_mem_mib)])
    import StudentModels            # installs the QKeras nest.is_sequence shim
    import partitions as P
    rep = dict(tensorflow=tf.__version__, gpus=[g.name for g in gpus],
               keras_module=tf.keras.__name__ if hasattr(tf.keras, '__name__') else str(tf.keras),
               TF_USE_LEGACY_KERAS=os.environ.get('TF_USE_LEGACY_KERAS'),
               TF_DETERMINISTIC_OPS=os.environ.get('TF_DETERMINISTIC_OPS'),
               TF_CUDNN_DETERMINISTIC=os.environ.get('TF_CUDNN_DETERMINISTIC'))

    # 1 parameter counts ---------------------------------------------------------------
    rep['params'] = {f'd{d}_u{u}_reset_after_{ra}': int(gru_model(tf, d, u, ra).count_params())
                     for d, u in ((5, 140), (7, 140), (9, 140), (9, 280)) for ra in (True, False)}
    print('[C3] params', rep['params'], flush=True)

    # 2 cuDNN use, speed, determinism at d=7 ---------------------------------------------
    d = 7
    tr = P.block('train')
    det = np.load(P.pool_path(d, 'formal'))
    x = StudentModels.to_sequence(det['det_evts'][tr[0]:tr[0] + N_TIME], d, d, P.load()['p'])
    y = det['flips'][tr[0]:tr[0] + N_TIME].astype(np.float32).reshape(-1, 1)
    del det
    rep['d7'] = {}
    for ra in (True, False):
        r = {}
        try:
            m = compiled(tf, gru_model(tf, d, UNITS, ra))
            m.fit(x[:3 * BATCH], y[:3 * BATCH], batch_size=BATCH, epochs=1, verbose=0)  # warm-up
            logdir = os.path.join(a.out, f'profile_reset_after_{ra}')
            tf.profiler.experimental.start(logdir)
            for i in range(3):
                m.train_on_batch(x[i * BATCH:(i + 1) * BATCH], y[i * BATCH:(i + 1) * BATCH])
            tf.profiler.experimental.stop()
            blob = b''.join(open(f, 'rb').read() for f in
                            glob.glob(os.path.join(logdir, '**', '*.xplane.pb'), recursive=True))
            r['cudnn_rnn_in_trace'] = b'CudnnRNN' in blob
            r['trace_bytes'] = len(blob)
            times = []

            class T(tf.keras.callbacks.Callback):
                def on_epoch_begin(self, e, logs=None): self.t = time.time()
                def on_epoch_end(self, e, logs=None): times.append(time.time() - self.t)
            m.fit(x, y, batch_size=BATCH, epochs=3, verbose=0, callbacks=[T()])
            steps = N_TIME // BATCH
            r['epoch_s'] = [round(t, 3) for t in times]
            r['steps_per_s'] = round(steps * 2 / sum(times[1:]), 1)   # skip the first epoch
            ws = []
            for _ in range(2):
                mm = compiled(tf, gru_model(tf, d, UNITS, ra, seed=123))
                tf.keras.utils.set_random_seed(123)
                mm.fit(x[:20 * BATCH], y[:20 * BATCH], batch_size=BATCH, epochs=1,
                       shuffle=True, verbose=0)
                ws.append(np.concatenate([w.ravel() for w in mm.get_weights()]))
            r['two_runs_bit_identical'] = bool(np.array_equal(ws[0], ws[1]))
            r['two_runs_max_abs_diff'] = float(np.max(np.abs(ws[0] - ws[1])))
        except Exception as exc:
            r['error'] = f'{type(exc).__name__}: {exc}'
            r['traceback'] = traceback.format_exc()[-2000:]
        rep['d7'][f'reset_after_{ra}'] = r
        print(f'[C3] d7 reset_after={ra}:', {k: v for k, v in r.items() if k != 'traceback'},
              flush=True)

    # 3 QKeras QGRU ----------------------------------------------------------------------
    rep['qgru'] = {}
    xs = x[:1000]
    try:
        from qkeras.qrecurrent import QGRU
        q = StudentModels._quantizers(8)
        KL = tf.keras.layers
        inp = KL.Input(shape=(d + 1, d * d - 1))
        h = QGRU(UNITS, return_sequences=False, reset_after=True, kernel_quantizer=q,
                 recurrent_quantizer=q, bias_quantizer=q, state_quantizer=q, name='gru')(inp)
        qm = tf.keras.Model(inp, StudentModels._dense(1, 'logit', 8)(h))
        out = qm.predict(xs, verbose=0)
        rep['qgru']['reset_after_True'] = dict(ok=bool(np.all(np.isfinite(out))),
                                               params=int(qm.count_params()),
                                               bias_shape=list(qm.get_layer('gru').cell.bias.shape))
    except Exception as exc:
        rep['qgru']['reset_after_True'] = dict(ok=False, error=f'{type(exc).__name__}: {exc}')
    try:
        qm = StudentModels.build_gru_student(d=d, rounds=d, units=UNITS, p=P.load()['p'],
                                             weight_bits=8)
        out = qm.predict(xs, verbose=0)
        rep['qgru']['reset_after_False_build_gru_student'] = dict(
            ok=bool(np.all(np.isfinite(out))), params=int(qm.count_params()),
            shim_is_sequence_present=hasattr(__import__('tensorflow.python.util.nest',
                                                        fromlist=['x']), 'is_sequence'))
    except Exception as exc:
        rep['qgru']['reset_after_False_build_gru_student'] = dict(
            ok=False, error=f'{type(exc).__name__}: {exc}')
    print('[C3] qgru', rep['qgru'], flush=True)

    with open(os.path.join(a.out, 'c3_results.json'), 'w') as fh:
        json.dump(rep, fh, indent=1)
    print(f"[C3] wrote {os.path.join(a.out, 'c3_results.json')}")


if __name__ == '__main__':
    main()
