#!/usr/bin/env python3
# from FNAL-QCDecoding-FPGA/eval_student_on_tail.py @ d5ee86f
# Created: 2026-08-05
# Last modified: 2026-10-02 (pool d/p/rounds checked against the arguments; --hidden default by student, as in train_student.py; CSV records the weights path and whether it is a .best checkpoint; the sealed block is scored only with --score-sealed, logged in configs/sealed_access.tsv, once per weights file unless --score-sealed-again; an existing CSV with an older header is left as it is and the row goes to <name>.v2.csv).
# 2026-10-05: --units takes several values for stacked GRU layers.
# 2026-10-06: pool rows read by range (pool_rows), so only the scored block is loaded; a block that a decoder already read (partitions spent_blocks) is flagged.
# Earlier 2026-09-26 (imports from helpers / circuit_generators; --reset-after; host-memory tf.data input; --unroll; --no-tf32)

"""Score a saved student on one block and run the paired McNemar test against MWPM.

MWPM is decoded on the same shots, never read from mwpm_baseline.csv.

python eval_student_on_tail.py \
    --weights ~/rcnn_threshold/out_student_r5_hard/r5hard_gru_ntr10000000_seed0_lr0.003.weights.h5 \
    --student gru --units 140 --d 5 --p 0.010 --rounds 5 \
    --pool ~/rcnn_threshold/pools_r5/data_d5_p0.010_r5_FORMAL.npz \
    --eval-start 20000000 --n-test 200000 \
    --out-csv ~/rcnn_threshold/student_mcnemar_r5.csv
"""
import argparse
import csv
import os
import numpy as np

# stim is imported only where MWPM is decoded, so the checks below are testable without it
from helpers import mcnemar_from_correct

LOG = '[student-eval]'

CSV_COLS = ['weights', 'student', 'd', 'p', 'rounds', 'eval_start', 'n_test',
            'n_params', 'p_L', 'mwpm_p_L', 'ratio', 'base_rate',
            'both_right', 'student_only', 'mwpm_only', 'both_wrong', 'n_discordant',
            'net_student_wins', 'mcnemar_chi2_cc', 'p_chi2', 'p_exact',
            'reset_after', 'unroll', 'tf32', 'weights_path', 'is_best_ckpt']


def sha256_of(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check_eval_block(args, hidden):
    # Checks before loading: files exist, pool d/p/rounds match, block fits, sealed block
    # untouched. The sealed block needs --score-sealed and is logged in
    # configs/sealed_access.tsv; a weights file already in the log also needs --score-sealed-again.
    import partitions as P
    if args.score_sealed_again and not args.score_sealed:
        raise SystemExit(f"{LOG} --score-sealed-again needs --score-sealed. STOP.")
    if not os.path.exists(args.pool):
        raise SystemExit(f"{LOG} MISSING pool {args.pool}")
    if not os.path.exists(args.weights):
        raise SystemExit(f"{LOG} MISSING weights {args.weights}")
    try:
        src = P.check_pool_geometry(args.pool, args.d, args.p, args.rounds)
    except ValueError as e:
        raise SystemExit(f"{LOG} {e}")
    if src is None:
        print(f"{LOG} WARNING: {os.path.basename(args.pool)} has no fingerprint and "
              f"no data_d*_p*_r* name; d/p/rounds not checked", flush=True)

    import pool_rows as PR
    n_pool = PR.n_rows(args.pool, 'flips')         # header only
    lo, hi = args.eval_start, args.eval_start + args.n_test
    if hi > n_pool:
        raise SystemExit(f"{LOG} partition [{lo}, {hi}) exceeds pool size {n_pool}")
    try:
        sealed, why = P.sealed_block(args.pool, n_pool)
    except ValueError as e:
        raise SystemExit(f"{LOG} {e}")
    if sealed is None:
        print(f"{LOG} WARNING: {why}", flush=True)
        return lo, hi
    if not (lo < sealed[1] and sealed[0] < hi):
        if args.score_sealed:
            print(f"{LOG} --score-sealed given, but [{lo:,}, {hi:,}) does not touch the "
                  f"sealed block; nothing logged", flush=True)
        return lo, hi
    for s0, s1, note in P.spent_blocks(args.pool):
        if lo < s1 and s0 < hi:
            print(f"{LOG} NOTE: [{s0:,}, {s1:,}) of this pool is {note}", flush=True)
    if not args.score_sealed:
        raise SystemExit(f"{LOG} block [{lo:,}, {hi:,}) overlaps the sealed block "
                         f"[{sealed[0]:,}, {sealed[1]:,}) of {os.path.basename(args.pool)} "
                         f"({why}). Scoring it needs --score-sealed. STOP.")
    sha = sha256_of(args.weights)
    earlier = P.sealed_accesses(sha)
    if earlier and not args.score_sealed_again:
        raise SystemExit(f"{LOG} these weights (sha256 {sha[:16]}) already scored the sealed "
                         f"block {len(earlier)} time(s), first at {earlier[0]['utc']} "
                         f"({P.sealed_log_path()}). A repeat needs --score-sealed-again. STOP.")
    import datetime
    log = P.record_sealed_access(dict(
        utc=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        weights_sha256=sha, weights_path=os.path.abspath(args.weights),
        student=args.student, units=('x'.join(map(str, args.units)) if args.student == 'gru' else ''),
        hidden='-'.join(str(h) for h in hidden), d=args.d, p=args.p, rounds=args.rounds,
        pool=os.path.abspath(args.pool), block_start=lo, block_stop=hi,
        repeat=int(bool(earlier))))
    print(f"{LOG} SEALED block [{lo:,}, {hi:,}) scored with --score-sealed"
          + (f" (repeat {len(earlier) + 1})" if earlier else "")
          + f"; logged in {log}", flush=True)
    return lo, hi


def append_eval_row(out_csv, row):
    # A file with an older header is left alone; the row goes to <name>.v2.csv instead.
    path = out_csv
    if os.path.exists(path):
        with open(path, newline='') as f:
            header = next(csv.reader(f), [])
        if header != CSV_COLS:
            path = os.path.splitext(out_csv)[0] + '.v2.csv'
            if os.path.exists(path):
                with open(path, newline='') as f:
                    if next(csv.reader(f), []) != CSV_COLS:
                        raise SystemExit(f"{LOG} {out_csv} and {path} both have headers "
                                         f"other than this script's. STOP.")
            print(f"{LOG} {out_csv} has an older header; row written to {path}", flush=True)
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(CSV_COLS)
        w.writerow(row)
    return path


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True,
                    help='.weights.h5 written by train_student.py')
    ap.add_argument('--student', choices=['mlp', 'gru'], required=True)
    ap.add_argument('--inputs', choices=['evts', 'evts+bits'], default='evts')
    ap.add_argument('--hidden', type=int, nargs='*', default=None,
                    help='MLP: the Dense stack (e.g. 209 209). GRU: post-GRU Dense layers. '
                         'Not given: GRU none, MLP 128 128 (train_student.py defaults). '
                         'Given with no values: none.')
    ap.add_argument('--units', type=int, nargs='+', default=[140],
                    help='GRU hidden state size; several values = stacked GRU layers')
    ap.add_argument('--reset-after', action='store_true',
                    help='the checkpoint was trained with reset_after=True')
    ap.add_argument('--unroll', action='store_true', help='build the GRU with unroll=True')
    ap.add_argument('--no-tf32', action='store_true',
                    help='disable TensorFloat-32 matmuls on the GPU (full FP32). Default: the TensorFlow default, TF32 on, as in every earlier run')
    ap.add_argument('--weight-bits', type=int, default=None)
    ap.add_argument('--act-bits', type=int, default=None)
    ap.add_argument('--d', type=int, default=5)
    ap.add_argument('--p', type=float, default=0.010)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--kernel', type=int, default=3)
    ap.add_argument('--pool', required=True)
    ap.add_argument('--eval-start', type=int, required=True,
                    help='first shot of the partition to score (e.g. 20000000)')
    ap.add_argument('--n-test', type=int, default=200000)
    ap.add_argument('--score-sealed', action='store_true',
                    help='allow a block overlapping the sealed block (configs/partitions.yaml). '
                         'Each such scoring is logged in configs/sealed_access.tsv with the '
                         'sha256 of the weights; without this flag the block is refused.')
    ap.add_argument('--score-sealed-again', action='store_true',
                    help='with --score-sealed: allow weights already in the sealed-access '
                         'log to score the sealed block again (the row is marked repeat=1)')
    ap.add_argument('--batch-size', type=int, default=10000)
    ap.add_argument('--out-csv', default=None)
    ap.add_argument('--dump-per-shot', default=None,
                    help='npz of per-shot arrays, for failure-case analysis')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--gpu-mem-mib', type=int, default=None,
                    help='cap this process to N MiB of GPU memory')
    args = ap.parse_args()

    # --hidden defaults as in train_student.py; checks run before TensorFlow is imported
    from train_student import resolve_hidden
    hidden = resolve_hidden(args.student, args.hidden)
    lo, hi = check_eval_block(args, hidden)

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    assert tf.__version__.startswith('2.15'), (
        f'need TF 2.15.x (Keras 2); got TF {tf.__version__}.')
    if args.no_tf32:
        tf.config.experimental.enable_tensor_float_32_execution(False)
    print(f"[student-eval] tf32={'off' if args.no_tf32 else 'on (TensorFlow default)'} "
          f"reset_after={args.reset_after} unroll={args.unroll}", flush=True)
    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')
    elif args.gpu_mem_mib:
        from train_student import cap_gpu_memory
        cap_gpu_memory(tf, args.gpu_mem_mib)

    from types_cfg import get_types
    from circuit_partition import split_measurements
    from StudentModels import build_student, build_gru_student, assemble_features, \
        student_pred_and_correct

    d, p, r, k = args.d, args.p, args.rounds, args.kernel
    binary_t, _t, idx_t, _pk = get_types(d, r, k)

    import pool_rows as PR
    te = slice(lo, hi)            # checked in check_eval_block; only these rows are read
    print(f"[student-eval] scoring shots [{lo:,}, {hi:,}) of {os.path.basename(args.pool)}",
          flush=True)

    m_te = PR.rows(args.pool, 'measurements', lo, hi).astype(binary_t)
    e_te = PR.rows(args.pool, 'det_evts', lo, hi).astype(binary_t)
    truth = PR.rows(args.pool, 'flips', lo, hi).astype(np.int8).reshape(-1)
    b_te, _, _ = split_measurements(m_te, d, idx_t)
    from host_data import to_host_uint8, make_dataset
    x_te = assemble_features(b_te, e_te, args.inputs, student=args.student,
                             d=d, rounds=r, p=p)
    det_evts_te = e_te            # kept for the MWPM decode below
    del m_te, b_te

    # same architecture as training; a units/hidden mismatch raises on load
    if args.student == 'gru':
        units = args.units[0] if len(args.units) == 1 else tuple(args.units)
        model = build_gru_student(d=d, rounds=r, inputs=args.inputs, units=units,
                                  hidden=hidden, p=p,
                                  weight_bits=args.weight_bits, act_bits=args.act_bits,
                                  reset_after=args.reset_after, unroll=args.unroll)
    else:
        model = build_student('mlp', d=d, rounds=r, inputs=args.inputs, hidden=hidden,
                              weight_bits=args.weight_bits, act_bits=args.act_bits)
    model.load_weights(args.weights)
    print(f"[student-eval] {args.student} params={model.count_params():,}", flush=True)

    # host-memory uint8 + tf.data (host_data.py), shot order preserved
    logits = model.predict(make_dataset(tf, to_host_uint8(x_te), None, args.batch_size),
                           verbose=0)
    s_pred, s_correct = student_pred_and_correct(logits, truth)
    pL = float((~s_correct).mean())
    base_rate = float(truth.mean())

    # MWPM decoded on these shots
    import pymatching
    from circuit_generators import build_circuit
    circ = build_circuit(d, p, r)
    dem = circ.detector_error_model(decompose_errors=True)
    pym = pymatching.Matching.from_detector_error_model(dem)
    m_pred = pym.decode_batch(det_evts_te, bit_packed_predictions=False,
                              bit_packed_shots=False).astype(np.int8).reshape(-1)
    m_correct = (m_pred == truth)
    mwpm_pL = float((~m_correct).mean())

    mc = mcnemar_from_correct(s_correct, m_correct)
    # b = student-only-correct
    print(f"[student-eval] student p_L={pL:.6f}  MWPM p_L={mwpm_pL:.6f}  "
          f"ratio={pL / mwpm_pL:.4f}x  base_rate={base_rate:.5f}", flush=True)
    print(f"[mcnemar] 2x2 on {args.n_test:,} shared shots:", flush=True)
    print(f"[mcnemar]   both-right   = {mc['both_right']:,}", flush=True)
    print(f"[mcnemar]   student-only = {mc['rcnn_only']:,}   (student wins these)", flush=True)
    print(f"[mcnemar]   MWPM-only    = {mc['mwpm_only']:,}   (MWPM wins these)", flush=True)
    print(f"[mcnemar]   both-wrong   = {mc['both_wrong']:,}", flush=True)
    print(f"[mcnemar]   net student wins = {mc['net_rcnn_wins']:+,}  "
          f"(= n_test x (p_L_MWPM - p_L_student) = {args.n_test * (mwpm_pL - pL):+.0f})",
          flush=True)
    print(f"[mcnemar]   chi2(cc)={mc['mcnemar_chi2_cc']:.1f}  p_chi2={mc['p_chi2']:.3e}  "
          f"p_exact={mc['p_exact']:.3e}", flush=True)

    if args.dump_per_shot:
        np.savez_compressed(
            args.dump_per_shot,
            shot_idx=np.arange(lo, hi, dtype=np.int64), truth=truth,
            student_pred=s_pred, student_correct=s_correct,
            student_logit=np.asarray(logits).reshape(-1).astype(np.float32),
            mwpm_pred=m_pred, mwpm_correct=m_correct,
            det_evts=det_evts_te.astype(np.int8),
            d=d, p=p, rounds=r, n_test=args.n_test)
        print(f"[student-eval] dumped per-shot -> {args.dump_per_shot}", flush=True)

    if args.out_csv:
        written = append_eval_row(args.out_csv, [
            os.path.basename(args.weights), args.student, d, p, r,
            lo, args.n_test, int(model.count_params()),
            round(pL, 6), round(mwpm_pL, 6), round(pL / mwpm_pL, 4),
            round(base_rate, 5),
            mc['both_right'], mc['rcnn_only'], mc['mwpm_only'],
            mc['both_wrong'], mc['n_discordant'], mc['net_rcnn_wins'],
            round(mc['mcnemar_chi2_cc'], 3), mc['p_chi2'], mc['p_exact'],
            int(args.reset_after), int(args.unroll),
            'off' if args.no_tf32 else 'on',
            os.path.abspath(args.weights),
            int(os.path.basename(args.weights).endswith(('.best.weights.h5', '.best.keras')))])
        print(f"[student-eval] appended -> {written}", flush=True)


if __name__ == '__main__':
    run()
