# Created: 2026-09-27
# Last modified: 2026-09-27
"""Write the reproduction data for the synthesis package (repo-side; needs our pools).

    python3 make_reproduce_data.py --package synthesis_package

Writes into <package>/reproduce/:
  eval_inputs_10000.npy                  first 10,000 shots of the d=5 evaluation block
                                         [15,200,000, 15,210,000), scattered to (6, 24),
                                         float32 0/1 - the same shots as the recorded checks
  keras_logits_reset_after_False.npy     Keras FP32 CPU logits of models/..._False.keras
  keras_logits_reset_after_True.npy      Keras FP32 CPU logits of models/..._True.keras
Evaluation-block shots only; the sealed partition is not read.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.expanduser('~/FNAL-AI4QC-GRU')
sys.path.insert(0, os.path.join(REPO, 'src'))
N = 10_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', required=True)
    a = ap.parse_args()
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import partitions as P
    from StudentModels import to_sequence
    e0, e1 = P.block('eval')
    assert e0 + N <= e1
    x = to_sequence(np.load(P.pool_path(5, 'formal'))['det_evts'][e0:e0 + N], 5, 5,
                    0.004).astype(np.float32)
    out = os.path.join(a.package, 'reproduce')
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, 'eval_inputs_10000.npy'), x)
    # The recorded checks (report.json of the PR #1568 run) used these exact shots; compare
    # the packaged models' logits with the Keras logits implied there as a consistency check.
    info = dict(shots=f'[{e0}, {e0 + N}) of the d=5 evaluation block', shape=list(x.shape))
    for ra in (False, True):
        m = tf.keras.models.load_model(
            os.path.join(a.package, 'models', f'gru_u140_reset_after_{ra}.keras'), compile=False)
        y = m.predict(x, batch_size=10000, verbose=0).reshape(-1).astype(np.float32)
        np.save(os.path.join(out, f'keras_logits_reset_after_{ra}.npy'), y)
        info[f'reset_after_{ra}'] = dict(logit_std=float(y.std()), frac_positive=float((y > 0).mean()))
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
