#!/usr/bin/env python3
# from FNAL-QCDecoding-FPGA/dump_teacher_probs.py @ d5ee86f
# Created: 2026-07-29
# Last modified: 2026-10-06 (pool arrays read by row range with pool_rows; pool_fingerprint takes a path and uses the pool's .fingerprint.json, so the sealed rows are never read).
# Earlier 2026-08-17

"""Cache the frozen teacher's per-shot output over a pool, for distillation.

Writes an .npz with p_teacher, logit_teacher, flips, shot_idx, plus metadata and fingerprints.
--teacher-arch rcnn has a sigmoid head (logit recovered in float64); --teacher-arch gru has a
linear head (output is the logit, p = sigmoid(logit)). No train/test split here.

python dump_teacher_probs.py \
    --weights ~/rcnn_threshold/out_t200k_w/rcnn_d5_p0.010_r3_seed0_ntr10000000.weights.h5 \
    --d 5 --p 0.010 --rounds 3 --n-shots 1000000 \
    --out ~/rcnn_threshold/teacher/teacher_seed0_first1M.npz
"""
import argparse
import hashlib
import os
import numpy as np


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def pool_fingerprint(pool_npz):
    # (flips_sha256, measurements shape); flips and measurements come from one draw
    if isinstance(pool_npz, str):
        # path: read from .fingerprint.json so no pool row is read; else stream flips
        import json
        import pool_rows as PR
        fp = pool_npz.replace('.npz', '.fingerprint.json')
        if os.path.exists(fp):
            m = json.load(open(fp))
            if m.get('flips_sha256') and m.get('measurements_shape'):
                return m['flips_sha256'], m['measurements_shape']
        n = PR.n_rows(pool_npz, 'flips')
        h = hashlib.sha256()
        for a in range(0, n, 10_000_000):
            h.update(np.ascontiguousarray(PR.rows(pool_npz, 'flips', a, min(a + 10_000_000, n))).tobytes())
        return h.hexdigest(), str((n,) + tuple(PR.rows(pool_npz, 'measurements', 0, 1).shape[1:]))
    flips = np.ascontiguousarray(pool_npz['flips'])
    digest = hashlib.sha256(flips.tobytes()).hexdigest()
    return digest, str(pool_npz['measurements'].shape)


def recover_logit(p):
    # p clipped off 0 and 1, so |z| <= ~16
    p64 = np.asarray(p, dtype=np.float64).reshape(-1)
    eps = 1e-7
    p64 = np.clip(p64, eps, 1.0 - eps)
    return np.log(p64 / (1.0 - p64))


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True,
                    help='teacher .weights.h5 saved by train_one.py --save-weights, or by '
                         'train_student.py when --teacher-arch gru')
    ap.add_argument('--teacher-arch', choices=['rcnn', 'gru'], default='rcnn',
                    help="'rcnn' = CNNModel.FullRCNNModel (sigmoid head, probability out); "
                         "'gru' = StudentModels.build_gru_student (linear head, logit out). "
                         'See the module docstring for why the two are converted differently.')
    ap.add_argument('--units', type=int, default=64,
                    help='GRU hidden-state width; --teacher-arch gru only. Must match the '
                         'width the checkpoint was trained with or load_weights fails.')
    ap.add_argument('--student-hidden', type=int, nargs='*', default=[],
                    help='Dense widths between the GRU and its head; --teacher-arch gru '
                         'only. Empty (the default) matches the studies run so far.')
    ap.add_argument('--d', type=int, required=True)
    ap.add_argument('--p', type=float, required=True)
    ap.add_argument('--rounds', type=int, required=True)
    ap.add_argument('--kernel', type=int, default=3)
    ap.add_argument('--hidden', type=int, default=100)
    ap.add_argument('--hidden-layers', type=int, default=2)
    ap.add_argument('--npol', type=int, default=2)
    ap.add_argument('--n-start', type=int, default=0,
                    help='first shot index in the pool to dump (default 0 = training prefix)')
    ap.add_argument('--n-shots', type=int, required=True,
                    help='number of shots to dump starting at --n-start')
    ap.add_argument('--weight-bits', type=int, default=None,
                    help='if set, rebuild the QUANTIZED teacher instead of FP32. Only needed '
                         'if you ever want to distil from a quantized teacher; the plan of '
                         'record distils from the FP32 teacher, so leave this unset.')
    ap.add_argument('--batch-size', type=int, default=10000)
    ap.add_argument('--data-dir', default=os.path.expanduser('~/rcnn_threshold/pools'))
    ap.add_argument('--pool', default=None,
                    help='explicit pool npz; overrides the --data-dir name convention')
    ap.add_argument('--out', required=True, help='output .npz path for the cached outputs')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    # Keras 3 fails deep inside CNNModel.py's custom layers; fail early instead
    assert tf.__version__.startswith('2.15'), (
        f'need TF 2.15.x (Keras 2); got TF {tf.__version__}. '
        f'Activate the uv venv: source .venv/bin/activate, or run with '
        f'PY=<repo>/.venv/bin/python.')
    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')

    from types_cfg import get_types
    from circuit_partition import split_measurements

    d, p, r, k = args.d, args.p, args.rounds, args.kernel
    binary_t, _time_t, idx_t, _packed_t = get_types(d, r, k)

    fn = args.pool or os.path.join(args.data_dir, f'data_d{d}_p{p:.3f}_r{r}.npz')
    if not os.path.exists(fn):
        raise SystemExit(f"[teacher] MISSING pool {fn}")
    if not os.path.exists(args.weights):
        raise SystemExit(f"[teacher] MISSING weights {args.weights}")

    import pool_rows as PR
    z = fn                                         # a path; rows are read by range below
    N = PR.n_rows(fn, 'measurements')

    # fingerprints re-checked by train_student.py: catch a regenerated pool or retrained
    # teacher under the same filename
    weights_sha = sha256_file(args.weights)
    pool_flips_sha, pool_meas_shape = pool_fingerprint(z)
    print(f"[teacher] weights sha256={weights_sha[:16]}...  "
          f"pool flips sha256={pool_flips_sha[:16]}...  measurements{pool_meas_shape}",
          flush=True)
    lo, hi = args.n_start, args.n_start + args.n_shots
    if hi > N:
        raise SystemExit(f"[teacher] requested shots [{lo}, {hi}) exceed pool size {N}")
    sl = slice(lo, hi)

    # slice first so only the requested shots are loaded
    det_evts = PR.rows(fn, 'det_evts', lo, hi).astype(binary_t)
    flips = PR.rows(fn, 'flips', lo, hi).astype(binary_t)

    hidden = [args.hidden for _ in range(args.hidden_layers)]
    if args.teacher_arch == 'gru':
        # GRU reads det_evts only; same scatter as the trainer
        from StudentModels import assemble_features, build_gru_student
        model = build_gru_student(d=d, rounds=r, inputs='evts', units=args.units,
                                  hidden=tuple(args.student_hidden), p=p)
        x = assemble_features(None, det_evts, 'evts', student='gru', d=d, rounds=r, p=p)
        print(f"[teacher] gru teacher units={args.units} "
              f"hidden={tuple(args.student_hidden)} params={model.count_params():,} "
              f"input{x.shape[1:]}", flush=True)
        model.load_weights(args.weights)

        # linear head: output is the logit
        pred = model.predict(x, batch_size=args.batch_size, verbose=0)
        logit_teacher = np.asarray(pred, dtype=np.float32).reshape(-1)
        p_teacher = (1.0 / (1.0 + np.exp(-logit_teacher.astype(np.float64)))).astype(np.float32)
        del x
    else:
        measurements = PR.rows(fn, 'measurements', lo, hi).astype(binary_t)
        det_bits, _obs_bits, _data_bits = split_measurements(measurements, d, idx_t)
        if args.weight_bits is None or args.weight_bits >= 32:
            from CNNModel import FullRCNNModel
            model = FullRCNNModel(
                'ZL', d, k, r, hidden, npol=args.npol, stop_round=None,
                has_nonuniform_response=False, do_all_data_qubits=False,
                return_all_rounds=False)
        else:
            from CNNModel_quantized import build_quantized_rcnn
            model = build_quantized_rcnn(
                args.weight_bits, 'ZL', d, k, r, hidden, npol=args.npol, stop_round=None,
                has_nonuniform_response=False, do_all_data_qubits=False,
                return_all_rounds=False)
        _ = model([det_bits[0:1], det_evts[0:1]])  # build so the checkpoint layout matches
        model.load_weights(args.weights)

        # sigmoid head: output is p
        pred = model.predict([det_bits, det_evts], batch_size=args.batch_size, verbose=0)
        p_teacher = np.asarray(pred, dtype=np.float32).reshape(-1)
        logit_teacher = recover_logit(p_teacher).astype(np.float32)
        del measurements, det_bits

    truth = np.asarray(flips, dtype=np.int8).reshape(-1)

    # sanity print only (in-sample when the range covers training): catches a broken load
    teacher_pred = (p_teacher > 0.5).astype(np.int8)
    pL_insample = float((teacher_pred != truth).mean())
    base_rate = float(truth.mean())
    frac_confident = float(((p_teacher < 0.05) | (p_teacher > 0.95)).mean())
    print(f"[teacher] shots [{lo:,}, {hi:,})  p_L(on these shots)={pL_insample:.5f}  "
          f"base_rate={base_rate:.5f}  mean_p={p_teacher.mean():.5f}  "
          f"frac_confident={frac_confident:.3f}", flush=True)
    if pL_insample >= base_rate:
        print("[teacher] WARNING: teacher does not beat the all-zero base rate on these "
              "shots -- check the weights path and the architecture args.", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    np.savez_compressed(
        args.out,
        p_teacher=p_teacher,
        logit_teacher=logit_teacher,
        flips=truth,
        shot_idx=np.arange(lo, hi, dtype=np.int64),
        # metadata
        d=d, p=p, rounds=r, kernel=k, hidden=args.hidden,
        hidden_layers=args.hidden_layers, npol=args.npol,
        teacher_arch=args.teacher_arch, units=args.units,
        student_hidden=np.asarray(args.student_hidden, dtype=np.int64),
        weight_bits=(-1 if args.weight_bits is None else args.weight_bits),
        n_start=lo, n_shots=args.n_shots,
        weights_path=os.path.abspath(args.weights),
        pool_path=os.path.abspath(fn),
        # fingerprints
        weights_sha256=weights_sha,
        pool_flips_sha256=pool_flips_sha,
        pool_measurements_shape=pool_meas_shape,
    )
    print(f"[teacher] wrote -> {args.out}", flush=True)


if __name__ == '__main__':
    run()
