# Created: 2026-10-18
# Last modified: 2026-10-18

"""Score the teacher on the eval block through the training-step path and compare shot by shot
with the eval_student_on_tail.py dump."""
import argparse
import hashlib
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src'))

ap = argparse.ArgumentParser()
ap.add_argument('--d', type=int, required=True)
ap.add_argument('--units', type=int, nargs='+', required=True)
ap.add_argument('--weights', required=True)
ap.add_argument('--params', type=int, required=True)
ap.add_argument('--epoch', type=int, required=True)
ap.add_argument('--pool', required=True)
ap.add_argument('--dump', required=True)
ap.add_argument('--n', type=int, required=True)
ap.add_argument('--eval-start', type=int, default=15_200_000)
ap.add_argument('--no-tf32', action='store_true')
ap.add_argument('--out-json', required=True)
args = ap.parse_args()
d, p = args.d, 0.004
EVAL_START = args.eval_start

with h5py.File(args.weights, 'r') as f:
    epoch = int(f['optimizer/vars/0'][()]) // 10_000
sha = hashlib.sha256(open(args.weights, 'rb').read()).hexdigest()

dump = np.load(args.dump)
if int(dump['shot_idx'][0]) != EVAL_START or len(dump['shot_idx']) < args.n:
    raise SystemExit(f"dump covers shots from {int(dump['shot_idx'][0])}, {len(dump['shot_idx'])} long")

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf
if args.no_tf32:
    tf.config.experimental.enable_tensor_float_32_execution(False)
import fresh_shots as FS
import pool_rows as PR
import train_student as TS
from StudentModels import detector_sequence_layout, to_sequence
from host_data import to_host_uint8

TS.cap_gpu_memory(tf, 8000)
teacher = TS.build_teacher_from_weights(tf, args.weights, args.units, d, d, p, args.params)

det = PR.rows(args.pool, 'det_evts', EVAL_START, EVAL_START + args.n).astype(np.uint8)
truth = PR.rows(args.pool, 'flips', EVAL_START, EVAL_START + args.n).astype(np.int8).reshape(-1)
_, _, imap = detector_sequence_layout(d, d, p)
x_train = FS._scatter(det, imap)
n_cmp = min(args.n, 100_000)
same_features = bool(np.array_equal(x_train[:n_cmp], to_host_uint8(to_sequence(det[:n_cmp], d, d, p))))
del det


@tf.function
def column(x, y):
    return TS.teacher_logit_column(tf, teacher, tf.cast(x, tf.float32), y)


z = np.empty(args.n, np.float32)
for a in range(0, args.n, 10_000):
    y = np.stack([truth[a:a + 10_000].astype(np.float32), np.zeros(10_000, np.float32)], axis=1)
    z[a:a + 10_000] = column(tf.constant(x_train[a:a + 10_000]), tf.constant(y)).numpy()[:, 1]

z_ref = dump['student_logit'][:args.n]
pred, pred_ref = (z > 0).astype(np.int8), (z_ref > 0).astype(np.int8)
errors = int((pred != truth).sum())
errors_ref = int((pred_ref != truth).sum())
out = dict(d=d, weights=os.path.abspath(args.weights), sha256=sha, params=int(teacher.count_params()),
           epoch=epoch, epoch_expected=args.epoch, tf32='off' if args.no_tf32 else 'on',
           n=args.n, features_equal=same_features, errors_training_path=errors,
           errors_eval_dump=errors_ref, decisions_differ=int((pred != pred_ref).sum()),
           max_abs_logit_diff=float(np.max(np.abs(z - z_ref))),
           labels_equal=bool(np.array_equal(truth, dump['truth'][:args.n].astype(np.int8))))
json.dump(out, open(args.out_json, 'w'), indent=1)
print(json.dumps(out, indent=1), flush=True)
