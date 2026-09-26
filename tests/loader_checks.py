# Created: 2026-09-26
# Last modified: 2026-09-26
"""Checks for the host-memory input pipeline (src/host_data.py), before relaunching training.

    python tests/loader_checks.py coverage
    python tests/loader_checks.py train --out <json> -- <train_student.py arguments>

coverage  (a) Builds the real training pipeline over a 123,457-row array whose target
              column holds each row's own index, runs three epochs, and checks that every
              epoch returns every index exactly once, that the x rows match the indexed
              host rows, and that consecutive epochs are in different orders.
train     Runs train_student.py in this process with the given arguments, then records
          the peak GPU memory TensorFlow allocated (tf.config.experimental.get_memory_info)
          and the run's history. Used for (b) determinism (two identical short runs,
          compared by the caller) and (c) peak GPU memory at d=5 and d=7, batch 10,000.
"""
import json
import os
import runpy
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, SRC)


def coverage():
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    from host_data import make_dataset, to_host_uint8
    n, batch = 123_457, 10_000
    rng = np.random.default_rng(0)
    x = to_host_uint8(rng.integers(0, 2, size=(n, 6, 24)))
    y = np.arange(n, dtype=np.float64)[:, None]          # target = own index, exact
    ds = make_dataset(tf, x, y, batch, shuffle_seed=7, to_gpu=False)
    orders, rows_ok = [], True
    for _ in range(3):
        idx = []
        for xb, yb in ds:
            i = yb.numpy()[:, 0].astype(np.int64)
            rows_ok &= bool(np.array_equal(xb.numpy(), x[i].astype(np.float32)))
            idx.append(i)
        orders.append(np.concatenate(idx))
    once = all(np.array_equal(np.sort(o), np.arange(n)) for o in orders)
    differ = not np.array_equal(orders[0], orders[1]) and not np.array_equal(orders[1], orders[2])
    ds2 = make_dataset(tf, x, y, batch, shuffle_seed=7, to_gpu=False)
    first2 = np.concatenate([yb.numpy()[:, 0].astype(np.int64) for _, yb in ds2])
    same_seed_same_order = bool(np.array_equal(first2, orders[0]))
    ok = once and rows_ok and differ and same_seed_same_order
    print(f"[loader] {'PASS' if ok else 'FAIL'} (a) coverage: every index once per epoch={once}, "
          f"rows match={rows_ok}, epochs reshuffled={differ}, same seed -> same first "
          f"epoch={same_seed_same_order}  (n={n:,}, 3 epochs, batch {batch:,})", flush=True)
    return ok


def train(out, args):
    sys.argv = [os.path.join(SRC, 'train_student.py')] + args
    os.chdir(SRC)
    try:
        runpy.run_path(os.path.join(SRC, 'train_student.py'), run_name='__main__')
        err = None
    except SystemExit as e:
        err = None if e.code in (None, 0) else f'SystemExit {e.code}'
    except Exception as e:                                   # report, don't hide
        err = f'{type(e).__name__}: {e}'
    import tensorflow as tf
    peak = None
    if tf.config.list_logical_devices('GPU'):
        peak = tf.config.experimental.get_memory_info('GPU:0')['peak'] / 2**20
    rep = dict(args=args, error=err, peak_gpu_mib=peak)
    with open(out, 'w') as fh:
        json.dump(rep, fh, indent=1)
    print(f"[loader] train: error={err} peak GPU memory {peak:.0f} MiB" if peak is not None
          else f"[loader] train: error={err} (no GPU)", flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'coverage':
        sys.exit(0 if coverage() else 1)
    if sys.argv[1] == 'train':
        k = sys.argv.index('--')
        train(sys.argv[sys.argv.index('--out') + 1], sys.argv[k + 1:])
