# Created: 2026-09-26
# Last modified: 2026-09-26

"""A7: smoke checks on one finished seed directory from gh200_run_student_scaling.sh.

loss_decreases, val_logged, best_json (recomputed from the history), keras_reload
(.best.keras gives the same logits as the rebuilt model + .best.weights.h5), eval_report.
Validation-block shots only.

python tests/a7_smoke_check.py <OUT>/seed0
"""
import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))


def report(name, ok, detail=''):
    print(f"[A7] {'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    return ok


def main(seeddir):
    import partitions as P
    ok = True
    hist_path = glob.glob(os.path.join(seeddir, 'runs', '*.history.json'))[0]
    tag = os.path.basename(hist_path)[:-len('.history.json')]
    h = json.load(open(hist_path))
    n_ep = len(h['loss'])
    ok &= report('loss_decreases', h['loss'][-1] < h['loss'][0],
                 f"loss {h['loss'][0]:.5f} -> {h['loss'][-1]:.5f} over {n_ep} epochs")
    logged = all(len(h.get(k, [])) == n_ep for k in ('val_loss', 'val_ler', 'val_errors'))
    ok &= report('val_logged', logged,
                 'val_ler per epoch: ' + ', '.join(f'{x:.5f}' for x in h.get('val_ler', [])))

    ck = os.path.join(seeddir, 'ckpt', tag)
    best = json.load(open(ck + '.best.json'))
    order = sorted(range(n_ep), key=lambda i: (h['val_errors'][i], h['val_loss'][i]))
    want_epoch = order[0] + 1
    ok &= report('best_json', best['best_epoch'] == want_epoch,
                 f"chosen epoch {best['best_epoch']} (recomputed {want_epoch}; last epoch "
                 f"{n_ep}); val_ler {best['val_ler']:.6f} = {best['val_errors']:,}/"
                 f"{best['val_n']:,} errors")

    import tensorflow as tf
    from StudentModels import build_gru_student, to_sequence
    m = __import__('re').search(r'_d(\d+)_p004_u(\d+)_', tag)
    d, units = int(m.group(1)), int(m.group(2))
    va = P.block('val')
    det = np.load(P.pool_path(d, 'formal'))['det_evts'][va[0]:va[0] + 10_000]
    x = to_sequence(det, d, d, P.load()['p'])
    full = tf.keras.models.load_model(ck + '.best.keras', compile=False)
    rebuilt = build_gru_student(d=d, rounds=d, units=units, p=P.load()['p'])
    rebuilt.load_weights(ck + '.best.weights.h5')
    a = full.predict(x, batch_size=10000, verbose=0)
    b = rebuilt.predict(x, batch_size=10000, verbose=0)
    ok &= report('keras_reload', np.array_equal(a, b),
                 f"max abs diff {float(np.max(np.abs(a - b))):.3g}, params {full.count_params():,}")

    rows = list(csv.DictReader(open(os.path.join(seeddir, 'eval_best_ckpt.csv'))))
    r = rows[-1]
    keys = list(r.keys())
    ok &= report('eval_report', len(rows) > 0, '; '.join(f'{k}={r[k]}' for k in keys
                                                         if any(s in k.lower() for s in
                                                                ('p_l', 'mwpm', 'only', 'disc', 'n_test', 'weights'))))
    print(f"[A7] {'ALL PASS' if ok else 'FAILED'}")
    return ok


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1]) else 1)
