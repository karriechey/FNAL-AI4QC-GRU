# Created: 2026-10-14
# Last modified: 2026-10-29 (pool d, p and basis checked before scoring). Earlier 2026-10-14

"""Score checkpoints on the selection block FORMAL [0, 10M) (protocol/selection_rule.md).

Appends one row per checkpoint to the scores CSV; MWPM is decoded once per pool and cached.
Epochs already in the CSV are skipped, so it can be rerun as new epochs are saved.

python score_selection.py --model f9w_p003 --d 9 --p 0.003 --units 280 280 280 \
    --pool <FORMAL.npz> --steps-per-epoch 10000 --out-csv selection_scores.csv <weights.h5> ...
"""
import argparse
import csv
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src'))

SEL_START, SEL_STOP = 0, 10_000_000
COLS = ['model', 'd', 'p', 'epoch', 'n', 'errors', 'p_L', 'mwpm_errors', 'mwpm_p_L', 'ratio',
        'weights', 'scored_utc']

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--d', type=int, required=True)
ap.add_argument('--p', type=float, required=True)
ap.add_argument('--units', type=int, nargs='+', required=True)
ap.add_argument('--pool', required=True)
ap.add_argument('--steps-per-epoch', type=int, required=True)
ap.add_argument('--out-csv', required=True)
ap.add_argument('--mwpm-cache', default=os.path.expanduser('~/selection_mwpm'))
ap.add_argument('--gpu-mem-mib', type=int, default=8000)
ap.add_argument('weights', nargs='+')
args = ap.parse_args()
d, p = args.d, args.p


def epoch_of(path):
    with h5py.File(path, 'r') as f:
        n = int(f['optimizer/vars/0'][()])
    if n % args.steps_per_epoch:
        raise SystemExit(f'{path}: {n} updates is not a whole number of epochs')
    return n // args.steps_per_epoch


# one row per (model, epoch): a .best.weights.h5 is a copy of one of the epoch files
done = set()
if os.path.exists(args.out_csv):
    done = {int(r['epoch']) for r in csv.DictReader(open(args.out_csv)) if r['model'] == args.model}
todo = {}
for path in args.weights:
    e = epoch_of(path)
    if e not in done:
        todo.setdefault(e, path)
if not todo:
    sys.exit(0)

import pool_rows as PR
import partitions as P
P.check_pool_geometry(args.pool, d, p, d)       # d, p and basis (QEC_BASIS) of the pool
truth = PR.rows(args.pool, 'flips', SEL_START, SEL_STOP).astype(np.int8).reshape(-1)
det = PR.rows(args.pool, 'det_evts', SEL_START, SEL_STOP)

os.makedirs(args.mwpm_cache, exist_ok=True)
cache = os.path.join(args.mwpm_cache, f'mwpm_sel_d{d}_p{p:.3f}_{os.path.basename(args.pool)}')
if os.path.exists(cache):
    m_correct = np.load(cache)['mwpm_correct']
else:
    import pymatching
    from circuit_generators import build_circuit
    dem = build_circuit(d, p, d).detector_error_model(decompose_errors=True)
    m_pred = pymatching.Matching.from_detector_error_model(dem).decode_batch(det)
    m_correct = m_pred.astype(np.int8).reshape(-1) == truth
    np.savez(cache, mwpm_correct=m_correct, start=SEL_START, stop=SEL_STOP)
mwpm_errors = int((~m_correct).sum())

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf
tf.config.experimental.enable_tensor_float_32_execution(False)
from train_student import cap_gpu_memory
cap_gpu_memory(tf, args.gpu_mem_mib)
from StudentModels import build_gru_student, to_sequence, student_pred_and_correct
from host_data import to_host_uint8, make_dataset

x = to_host_uint8(to_sequence(det, d, d, p))
del det
units = args.units[0] if len(args.units) == 1 else tuple(args.units)
model = build_gru_student(d=d, rounds=d, inputs='evts', units=units, hidden=(), p=p)

new = not os.path.exists(args.out_csv)
with open(args.out_csv, 'a', newline='') as fh:
    w = csv.writer(fh)
    if new:
        w.writerow(COLS)
    for epoch, path in sorted(todo.items()):
        model.load_weights(path)
        logits = model.predict(make_dataset(tf, x, None, 10000), verbose=0)
        _, correct = student_pred_and_correct(logits, truth)
        errors = int((~correct).sum())
        n = len(truth)
        w.writerow([args.model, d, p, epoch, n, errors, round(errors / n, 7), mwpm_errors,
                    round(mwpm_errors / n, 7), round(errors / mwpm_errors, 4),
                    os.path.abspath(path), time.strftime('%FT%TZ', time.gmtime())])
        fh.flush()
        print(f'{args.model} epoch {epoch}: {errors} errors, {errors / mwpm_errors:.4f}x MWPM',
              flush=True)
