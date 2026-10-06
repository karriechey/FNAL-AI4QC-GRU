#!/usr/bin/env python3
# from FNAL-QCDecoding-FPGA/train_student.py @ d5ee86f
# Created: 2026-07-29
# Last modified: 2026-10-03 (--lr-warmup-epochs: linear warmup before the cosine; --ema-decay: moving average of the weights, which are the ones checkpointed and scored; --warm-start: initialise from another run's .best.keras, tensors of matching name and shape; --max-params: refuse a model over a parameter budget). 2026-10-05: --units takes several values for stacked GRU layers; one value as before. 2026-10-06: --teacher-model: soft targets computed in this process from a saved GRU teacher, on the exact training array (primary prefix and every extension pool), the validation block and the scored block. Earlier 2026-10-02 (guards: alpha<1 with --extra-train-pool refused; --eval-start or --test-pool required; train/validation/scored blocks checked disjoint per pool and against the sealed block of configs/partitions.yaml; pool d/p/rounds checked; --hidden default by student; auto tag built before the checkpoint prefix and names units, hidden, lr, schedule, epochs, reset_after; scored_weights column). Earlier 2026-10-01 (training set built in one preallocated uint8 array, every pool converted in chunks; no full-size intermediate copies). Earlier 2026-09-30 (training-only pools in the compact format: bit-packed detection events, no measurements). Earlier 2026-09-29 (--resume-weights, --initial-epoch, --resume-best-json, --resume-history: continue a finished run to more epochs). Earlier 2026-09-27 (imports from helpers; best checkpoint chosen by val LER, saved as full .keras; --reset-after; host-memory tf.data input; --unroll; --no-tf32; several --extra-train-pool; --lr-schedule cosine, --lr-end, --global-clipnorm, --log-grad-norm, --stop-after-steps)
"""Train one distillation student (MLP or GRU) against the cached FP32 teacher outputs.

A single point is (student, inputs, seed, n_train, alpha, temperature, weight_bits,
act_bits). This is the Phase-2 workhorse: it trains an hls4ml-native student to imitate
the custom RCNN teacher, so the FPGA flow never has to parse the teacher's unsupported
combiner ops (pow/sqrt/log/gather).

Hard-vs-soft is one flag:
  --alpha 1.0  pure hard labels, the control arm -- what the architecture does alone
  --alpha 0.0  pure distillation, learning only from the teacher's soft output
  0 < a < 1    a blend
One code path for all three, so the comparison is not confounded by two trainers.

The hard label is one bit. The teacher's output is a calibrated probability that also
says how ambiguous the syndrome was; a teacher near 0.5 is telling the student the shot
is a coin flip, which the hard label cannot express. That is what --alpha measures.

Temperature acts in logit space: the soft target is sigmoid(z_teacher / T) and the
student's logit is divided by T. The soft term carries a T^2 factor so its gradient stays
comparable to the hard term as T varies, otherwise a temperature scan doubles as a
loss-weight scan. T=1 is the untempered teacher.

Disjointness follows train_one_quantized.py: the tail comes from --test-pool when given,
otherwise from the same pool behind the `ntr <= N - nte` assert. The teacher cache is
also checked to cover the training range and to come from the same pool.

Reads  the pool npz (measurements, det_evts, flips) + the teacher cache from
       dump_teacher_probs.py.
Writes {out-dir}/student_{tag}.csv, .history.json, and .weights.h5.
"""
import argparse
import csv
import json
import os
import time
import numpy as np

from helpers import (set_seeds, learning_rate_scheduler, lookup_mwpm, warmup_cosine_lr_schedule,
                     make_adam_logging_grad_norm)
from host_data import to_host_uint8, make_dataset  # reuse the recipe

# Prefix on every log line; set to '[mlp]' or '[gru]' once args are parsed, so a saved log
# names the architecture that produced it. Module-level so helper functions can use it
# without an args parameter.
LOG = '[student]'


def sha16(path):
    """First 16 hex chars of a file's SHA-256, or '' if it is not there."""
    import hashlib
    if not path or not os.path.exists(path):
        return ''
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()[:16]


def run_provenance(args, train_pool, teacher_cache_npz):
    """Identity of every input that determined this run, written into the result CSV so a
    row stays traceable without relying on directory names: training pool and its flips
    fingerprint, tail pool, teacher cache and checkpoint hash, this script's hash, UTC.
    """
    import datetime
    prov = {
        'run_utc': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'train_pool': os.path.basename(train_pool),
        'train_pool_dir': os.path.basename(os.path.dirname(os.path.abspath(train_pool))),
        'test_pool': os.path.basename(args.test_pool) if args.test_pool else 'same-pool-tail',
        'teacher_cache': os.path.basename(args.teacher_cache) if args.teacher_cache else '',
        'teacher_weights_sha': '',
        'train_pool_flips_sha': '',
        'code_sha': sha16(os.path.abspath(__file__)),
    }
    if teacher_cache_npz is not None:
        for key, col in (('weights_sha256', 'teacher_weights_sha'),
                         ('pool_flips_sha256', 'train_pool_flips_sha')):
            if key in teacher_cache_npz.files:
                prov[col] = str(teacher_cache_npz[key])[:16]
    return prov


PROVENANCE_COLS = ['run_utc', 'train_pool', 'train_pool_dir', 'test_pool', 'teacher_cache',
                   'teacher_weights_sha', 'train_pool_flips_sha', 'code_sha']


def make_distillation_loss(alpha, temperature):
    """Blended hard-label and teacher-imitation loss.

    y_true is packed as two columns, [hard_label, teacher_logit], since Keras passes one
    target tensor. y_pred is the student's raw logit. A closure rather than a Loss
    subclass, so reloading weights needs no custom-object registration.
    """
    import tensorflow as tf
    bce = tf.keras.losses.binary_crossentropy
    T = float(temperature)
    a = float(alpha)

    def loss(y_true, y_pred):
        y_hard = y_true[:, 0:1]
        z_teacher = y_true[:, 1:2]
        z_student = y_pred

        # Hard term: ordinary BCE against the true logical flip, on logits.
        hard = bce(y_hard, z_student, from_logits=True)

        # Soft term: match the temperature-softened teacher distribution. Both sides are
        # divided by T; the T^2 factor restores the gradient scale (the 1/T inside the
        # softened sigmoid otherwise shrinks the gradient by 1/T^2).
        soft_target = tf.sigmoid(z_teacher / T)
        soft = bce(soft_target, z_student / T, from_logits=True) * (T * T)

        return a * hard + (1.0 - a) * soft

    return loss


class TailDiagnostics:
    """Per-epoch tail metrics, built as a Keras callback at call time.

    A student drifting toward one class shows a collapsing logit spread before p_L moves,
    so logit_std is the early warning for a too-hot learning rate.

    Per epoch:
      tail_p_L         error rate on the evaluation tail
      pred_pos_rate    fraction predicted positive (logit > 0)
      base_rate        fraction actually positive, logged every epoch so the comparison
                       needs no second file
      collapsed        1 when pred_pos_rate sits within eps of base_rate and the logit
                       spread is degenerate
      logit_mean/std   distribution of the raw student output
    """

    def __init__(self, x_tail, y_tail, batch_size, eps=0.01, logit_std_floor=0.05):
        self.x, self.y = x_tail, np.asarray(y_tail).reshape(-1)
        self.batch_size = batch_size
        self.eps = eps
        self.logit_std_floor = logit_std_floor
        self.base_rate = float(self.y.mean())

    def build(self):
        import tensorflow as tf
        outer = self

        class _CB(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs if logs is not None else {}
                kw = {} if hasattr(outer.x, 'element_spec') else dict(batch_size=outer.batch_size)
                logits = self.model.predict(outer.x, verbose=0, **kw).reshape(-1)
                pred = (logits > 0.0).astype(np.int8)
                p_L = float((pred != outer.y).mean())
                ppr = float(pred.mean())
                lmean, lstd = float(logits.mean()), float(logits.std())
                collapsed = int(abs(ppr - outer.base_rate) < outer.eps
                                and lstd < outer.logit_std_floor)
                logs['tail_p_L'] = p_L
                logs['pred_pos_rate'] = ppr
                logs['base_rate'] = outer.base_rate
                logs['collapsed'] = collapsed
                logs['logit_mean'] = lmean
                logs['logit_std'] = lstd
                print(f"    [tail] epoch {epoch + 1:3d}  p_L={p_L:.5f}  "
                      f"pred_pos={ppr:.4f} (base={outer.base_rate:.4f})  "
                      f"logit mean={lmean:+.3f} std={lstd:.3f}"
                      f"{'  <-- collapsed' if collapsed else ''}", flush=True)

        return _CB()


def teacher_agreement(student_pred, teacher_prob, truth, band):
    """Student-vs-teacher decision agreement, overall and restricted to ambiguous shots.

    The teacher is confident on ~80% of shots (p<0.05 or p>0.95), where almost any student
    agrees, so the raw rate is dominated by the easy bulk and reads high regardless: a
    student that coin-flips inside the band still scores ~0.90 raw.

    The informative figure is agreement where the teacher is unsure
    (band[0] < p_teacher < band[1]) -- the contested syndromes distillation should transfer.

    Returns the raw rate, the band-restricted rate, the confident-subset rate, the band
    population, and both models' error rates inside the band (agreeing there does not mean
    either is right).
    """
    p_t = np.asarray(teacher_prob).reshape(-1)
    s_pred = np.asarray(student_pred).reshape(-1).astype(np.int8)
    t_pred = (p_t > 0.5).astype(np.int8)
    truth = np.asarray(truth).reshape(-1).astype(np.int8)

    lo, hi = band
    ambiguous = (p_t > lo) & (p_t < hi)
    confident = ~ambiguous
    n_amb = int(ambiguous.sum())

    out = {
        'agree_all': round(float((s_pred == t_pred).mean()), 5),
        'n_ambiguous': n_amb,
        'frac_ambiguous': round(float(ambiguous.mean()), 5),
        'teacher_tail_p_L': round(float((t_pred != truth).mean()), 6),
    }
    out['agree_confident'] = (round(float((s_pred[confident] == t_pred[confident]).mean()), 5)
                              if confident.any() else '')
    if n_amb:
        out['agree_ambiguous'] = round(float((s_pred[ambiguous] == t_pred[ambiguous]).mean()), 5)
        out['student_p_L_ambiguous'] = round(float((s_pred[ambiguous] != truth[ambiguous]).mean()), 5)
        out['teacher_p_L_ambiguous'] = round(float((t_pred[ambiguous] != truth[ambiguous]).mean()), 5)
    else:
        out['agree_ambiguous'] = ''
        out['student_p_L_ambiguous'] = ''
        out['teacher_p_L_ambiguous'] = ''
    return out


def make_hard_accuracy():
    """Accuracy against the hard label only, ignoring the packed teacher column.

    Keras's stock accuracy would compare the logit against a 2-column target and report a
    meaningless number.
    """
    import tensorflow as tf

    def hard_accuracy(y_true, y_pred):
        y_hard = y_true[:, 0:1]
        pred = tf.cast(y_pred > 0.0, y_hard.dtype)  # logit > 0 == prob > 0.5
        return tf.reduce_mean(tf.cast(tf.equal(pred, y_hard), tf.float32))

    return hard_accuracy


def verify_cache_fingerprints(tc, pool_npz):
    """Check the cache's stored fingerprints against the pool and teacher on disk.

    Complements the runtime flips-agreement check, which only sees the shots the cache
    carries. These catch a pool regenerated under the same filename or a checkpoint
    retrained in place. A missing fingerprint warns rather than fails, so older caches
    stay usable.
    """
    from dump_teacher_probs import sha256_file, pool_fingerprint

    if 'pool_flips_sha256' not in tc.files:
        print(f"{LOG} warning: cache predates fingerprinting -- identity not verified. "
              "Re-run dump_teacher_probs.py to get the check.", flush=True)
        return

    flips_sha, meas_shape = pool_fingerprint(pool_npz)
    if flips_sha != str(tc['pool_flips_sha256']):
        raise SystemExit(
            f"{LOG} pool fingerprint mismatch -- the pool on disk is not the one the "
            "teacher cache was built from (same path, different content: regenerated "
            "pool?). Re-dump the teacher cache.")
    if meas_shape != str(tc['pool_measurements_shape']):
        raise SystemExit(
            f"{LOG} pool measurements shape {meas_shape} != cache's "
            f"{str(tc['pool_measurements_shape'])}.")

    # Only checkable if the checkpoint is still at the recorded path; a cache pulled from
    # EAF will point somewhere that does not exist locally, which is expected.
    wpath = str(tc['weights_path']) if 'weights_path' in tc.files else ''
    if 'weights_sha256' in tc.files and wpath and os.path.exists(wpath):
        if sha256_file(wpath) != str(tc['weights_sha256']):
            raise SystemExit(
                f"{LOG} teacher checkpoint mismatch -- the .weights.h5 at the cached "
                "path has changed since the cache was built (retrained in place?). The "
                "soft targets no longer come from these weights. Re-dump.")
        print(f"{LOG} fingerprints verified: pool and teacher checkpoint both match "
              "the cache.", flush=True)
    else:
        print(f"{LOG} pool fingerprint verified; teacher checkpoint not present "
              "locally, so its hash was not re-checked.", flush=True)


def load_teacher_cache(path, pool_path, lo, hi, pool_npz=None):
    """Load cached teacher outputs and verify they cover [lo, hi) of this pool.

    Every failure this checks for is silent otherwise: a cache built from a different
    pool, or covering a shorter range than the requested training prefix, would still
    broadcast into a training run and produce a plausible-looking loss curve fit against
    the wrong shots. Fail loudly instead.
    """
    if not os.path.exists(path):
        raise SystemExit(f"{LOG} MISSING teacher cache {path} -- run dump_teacher_probs.py")
    tc = np.load(path, allow_pickle=False)

    cached_pool = str(tc['pool_path']) if 'pool_path' in tc.files else ''
    if cached_pool and os.path.abspath(cached_pool) != os.path.abspath(pool_path):
        raise SystemExit(
            f"{LOG} teacher cache was built from a DIFFERENT pool:\n"
            f"  cache pool: {cached_pool}\n  train pool: {os.path.abspath(pool_path)}")

    if pool_npz is not None:
        verify_cache_fingerprints(tc, pool_npz)

    idx = tc['shot_idx']
    if idx[0] > lo or idx[-1] < hi - 1:
        raise SystemExit(
            f"{LOG} teacher cache covers shots [{int(idx[0])}, {int(idx[-1]) + 1}) but "
            f"training needs [{lo}, {hi}). Re-dump with a wider --n-shots.")
    # Map the requested absolute shot range onto positions within the cache.
    off = lo - int(idx[0])
    n = hi - lo
    return (tc['logit_teacher'][off:off + n].astype(np.float32),
            tc['p_teacher'][off:off + n].astype(np.float32),
            tc['flips'][off:off + n].astype(np.int8))




def _primary_gen_seed(pool_path):
    """Generation seed of a pool, from its fingerprint, or None when absent."""
    fp = pool_path.replace('.npz', '.fingerprint.json')
    if not os.path.exists(fp):
        return None
    try:
        return json.load(open(fp)).get('gen_seed')
    except (ValueError, OSError):
        return None


def cap_gpu_memory(tf, mib):
    """Cap this process's GPU memory so several trainings share one card predictably.

    TensorFlow's allocator grows into whatever is free and does not release it, so an
    unconstrained process on an idle card parks far more than it needs and a later process
    fails its first allocation. A logical-device limit fixes each process's share.
    Must run before any tensor is placed on the device.
    """
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        tf.config.set_logical_device_configuration(
            g, [tf.config.LogicalDeviceConfiguration(memory_limit=mib)])
    print(f"[gpu] per-process limit {mib} MiB on {len(gpus)} device(s)", flush=True)


def load_extension_features(zex, take, d, r, p, inputs, student, idx_t, binary_t,
                            chunk=2_000_000, out=None):
    """Features and labels of the first `take` shots of a training-only pool.

    Two file formats:
      full    : measurements, det_evts, flips (as written by sample_pool)
      compact : det_evts_packed (np.packbits along the detector axis), n_det, flips
                (as written by sample_pool_compact; same shots for the same seed)
    The GRU's input is built from the detection events only, so both formats give the
    same features (tests/compact_pool_check.sh checks this tensor for tensor). The compact
    format has no measurements, so it serves the GRU with inputs='evts' only.

    The features are built `chunk` shots at a time straight into a preallocated uint8
    array. assemble_features returns float32, 4 bytes per value, so converting a whole
    50M-shot pool at once held a float32 copy about 4x the size of the final array; per
    chunk the values are the same and the float32 copy stays small.

    out: optional uint8 array of shape (take, ...) to fill in place (a slice of the
    preallocated training array), so no separate copy of these features is made. Also
    used for the primary pool's training prefix (full format, shots [0, take)).
    """
    from circuit_partition import split_measurements
    from StudentModels import assemble_features
    f_ex = zex['flips'][0:take].astype(binary_t).reshape(-1)
    compact = 'det_evts_packed' in zex.files
    if compact:
        if not (student == 'gru' and inputs == 'evts'):
            raise SystemExit(f"{LOG} compact pool (detection events only) serves the GRU "
                             f"with inputs='evts' only; got student={student} inputs={inputs}")
        n_det = int(zex['n_det'])
        src = zex['det_evts_packed'][0:take]
    else:
        src_m = zex['measurements'][0:take]
        src = zex['det_evts'][0:take]
    x_ex = out
    for a in range(0, take, chunk):
        b = min(a + chunk, take)
        if compact:
            e_c = np.unpackbits(src[a:b], axis=1, count=n_det).astype(binary_t)
            b_c = None
        else:
            e_c = src[a:b].astype(binary_t)
            b_c, _, _ = split_measurements(src_m[a:b].astype(binary_t), d, idx_t)
        x_c = to_host_uint8(assemble_features(b_c, e_c, inputs, student=student,
                                              d=d, rounds=r, p=p))
        if x_ex is None:
            x_ex = np.empty((take,) + x_c.shape[1:], dtype=np.uint8)
        x_ex[a:b] = x_c
        del e_c, b_c, x_c
    return x_ex, f_ex


def resolve_hidden(student, hidden):
    """Dense widths for the student. None (flag not given) means the architecture's default:
    () for the GRU, (128, 128) for the MLP. A bare --hidden gives [] = no Dense layers; given
    values are used as they are."""
    if hidden is None:
        return () if student == 'gru' else (128, 128)
    return tuple(hidden)


def check_pools_and_partitions(args, train_fn):
    """Checks on the pools and blocks of this run, before anything is loaded or built.

    1. Every pool (primary, --test-pool, each --extra-train-pool) has the d, p, rounds of
       the arguments, read from its .fingerprint.json or else its file name; a pool with
       neither is warned about.
    2. The blocks this run reads, as (pool, start, stop): the training prefix of the
       primary pool, each extension's [0, take), the validation block, the scored block
       (both read from --test-pool when it is given, else from the primary pool), and the
       sealed block of every FORMAL pool among them (configs/partitions.yaml, cross-checked
       against the fingerprint). No two blocks of the same file may overlap.
    """
    import partitions as P
    pools = ([('primary', train_fn)] + ([('test', args.test_pool)] if args.test_pool else [])
             + [('extension', x) for x in (args.extra_train_pool or [])])
    for role, pool in pools:
        if not os.path.exists(pool):
            raise SystemExit(f"{LOG} MISSING {role} pool {pool}")
        try:
            src = P.check_pool_geometry(pool, args.d, args.p, args.rounds)
        except ValueError as e:
            raise SystemExit(f"{LOG} {role} pool: {e}")
        if src is None:
            print(f"{LOG} WARNING: {role} pool {os.path.basename(pool)} has no fingerprint "
                  f"and no data_d*_p*_r* name; d/p/rounds not checked", flush=True)

    sizes = {}

    def n_of(pool):
        if pool not in sizes:
            sizes[pool] = int(np.load(pool)['flips'].shape[0])
        return sizes[pool]

    score_pool = args.test_pool or train_fn
    e0 = (n_of(score_pool) - args.n_test) if args.eval_start is None else args.eval_start
    blocks = [('training prefix', train_fn, 0, args.n_train),
              ('scored block', score_pool, e0, e0 + args.n_test)]
    if args.val_start is not None and args.val_n is not None:
        # Validation is read from the same file as the scored block (see the loader below).
        blocks.append(('validation', score_pool, args.val_start, args.val_start + args.val_n))
    counts = list(args.extra_train_n or [])
    for i, ext in enumerate(args.extra_train_pool or []):
        n_ex = n_of(ext)
        take = min((counts[i] if i < len(counts) else None) or n_ex, n_ex)
        blocks.append(('training extension', ext, 0, take))
    for pool in dict.fromkeys(b[1] for b in blocks):
        try:
            sealed, why = P.sealed_block(pool, n_of(pool))
        except ValueError as e:
            raise SystemExit(f"{LOG} {e}")
        if sealed is not None:
            blocks.append(('sealed block', pool, sealed[0], sealed[1]))
            print(f"{LOG} sealed block of {os.path.basename(pool)}: "
                  f"[{sealed[0]:,}, {sealed[1]:,}) ({why}); not read", flush=True)
        elif not why.startswith('training-only'):
            print(f"{LOG} WARNING: {why}", flush=True)
    try:
        P.check_disjoint(blocks)
    except ValueError as e:
        raise SystemExit(f"{LOG} {e}")
    where = (f"--test-pool {os.path.basename(args.test_pool)}" if args.test_pool
             else f"the primary pool {os.path.basename(train_fn)}")
    print(f"{LOG} blocks disjoint: training prefix [0, {args.n_train:,}) of the primary pool; "
          + (f"validation [{args.val_start:,}, {args.val_start + args.val_n:,}) and "
             if args.val_start is not None and args.val_n is not None else "")
          + f"scored [{e0:,}, {e0 + args.n_test:,}) read from {where}", flush=True)


def run():
    ap = argparse.ArgumentParser()
    # --- problem config (must match the teacher's) ---
    ap.add_argument('--d', type=int, default=5)
    ap.add_argument('--p', type=float, default=0.010)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--kernel', type=int, default=3)
    # --- student architecture ---
    ap.add_argument('--student', choices=['mlp', 'gru'], required=True)
    ap.add_argument('--inputs', choices=['evts', 'evts+bits'], default='evts',
                    help="'evts' = detector events only (what MWPM sees, smallest FPGA "
                         "input). 'evts+bits' also feeds the raw stabilizer measurements, "
                         "which the teacher sees.")
    ap.add_argument('--hidden', type=int, nargs='*', default=None,
                    help='MLP: the Dense stack. GRU: optional Dense layers after the GRU. '
                         'Not given: GRU none, MLP 128 128. Given with no values: none.')
    ap.add_argument('--units', type=int, nargs='+', default=[64],
                    help='GRU hidden state size; several values = stacked GRU layers '
                         '(e.g. 280 280). One value builds the single-layer model as before.')
    ap.add_argument('--reset-after', action='store_true',
                    help='GRU with reset_after=True (cuDNN form); default False as before')
    ap.add_argument('--unroll', action='store_true',
                    help='unroll the GRU timesteps (generic kernel, no cuDNN); default False')
    ap.add_argument('--no-tf32', action='store_true',
                    help='disable TensorFloat-32 matmuls on the GPU (full FP32). Default: the TensorFlow default, TF32 on, as in every earlier run')
    ap.add_argument('--weight-bits', type=int, default=None,
                    help='Phase-3 QAT weight word length; None/>=32 => float weights '
                         '(the Phase-2 distillation setting)')
    ap.add_argument('--act-bits', type=int, default=None,
                    help='Phase-3 activation word length; None/>=32 => float activations')
    # --- distillation ---
    ap.add_argument('--teacher-cache', default=None,
                    help='npz from dump_teacher_probs.py over the TRAINING prefix. '
                         'Required unless --alpha 1.0.')
    ap.add_argument('--teacher-model', default=None,
                    help='a .keras GRU teacher (logit head, e.g. a .best.keras). Its logits are '
                         'computed here, in this process, on the training array as built '
                         '(primary prefix and every --extra-train-pool shot), on the '
                         'validation block and on the scored block, so the soft targets are '
                         'aligned with the training shots by construction. Replaces '
                         '--teacher-cache / --val-teacher-cache / --teacher-tail-cache and '
                         'allows alpha < 1 with extension pools.')
    ap.add_argument('--teacher-tail-cache', default=None,
                    help='npz from dump_teacher_probs.py over the EVALUATION TAIL. '
                         'Optional; enables the student-vs-teacher agreement columns, '
                         'which are the distillation gap measurement.')
    ap.add_argument('--ambiguous-band', type=float, nargs=2, default=[0.05, 0.95],
                    metavar=('LO', 'HI'),
                    help='teacher-probability band counted as ambiguous. Agreement is '
                         'reported both raw and restricted to this band. The raw rate is '
                         'dominated by the ~80%% of shots where this teacher is confident '
                         'and reads misleadingly high; the band-restricted rate is the '
                         'one that measures whether the teacher\'s decision function '
                         'actually transferred. Default 0.05-0.95.')
    ap.add_argument('--alpha', type=float, default=0.0,
                    help='weight on the hard-label term. 1.0 = hard labels only (control '
                         'arm, needs no teacher cache); 0.0 = pure distillation.')
    ap.add_argument('--temperature', type=float, default=1.0)
    # --- training recipe (mirrors train_one.py) ---
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--n-train', type=int, required=True)
    ap.add_argument('--n-test', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=10000)
    ap.add_argument('--val-split', type=float, default=0.2,
                    help='fallback only, used when --val-start/--val-n are not given. '
                         'Carves this fraction OUT of the training prefix, so n_train '
                         'overstates the shots that actually produce gradients.')
    # Explicit validation block, mirroring train_one.py's --val-start/--val-n. Without
    # these Keras carves validation_split out of the training prefix, so a student at
    # n_train=N trains on 0.8N while the teacher at the same n_train trains on N -- the
    # two ladders are then not comparable on the x-axis, which is the whole point of
    # plotting them together.
    ap.add_argument('--val-start', type=int, default=None,
                    help='first shot of an explicit validation block. Set with --val-n to '
                         'pass validation_data and disable validation_split, so the full '
                         'training prefix produces gradients.')
    ap.add_argument('--val-n', type=int, default=None)
    ap.add_argument('--val-teacher-cache', default=None,
                    help='teacher cache covering the validation block. Required with '
                         '--val-start when alpha < 1: the soft term of the loss needs '
                         'teacher logits on the validation shots too, and the training '
                         'cache does not cover them (it stops at n_train). Usually the '
                         'same npz passed to --teacher-tail-cache, since validation and '
                         'the scored partition are normally the same shots.')
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--no-early-stopping', action='store_true')
    # --- best-validation checkpointing (added 2026-08-06 for the parameter-reduction
    # study, which selects a checkpoint on one slice and reports on a disjoint one).
    # Both default to None, so every earlier student run reproduces unchanged: with no
    # --ckpt-dir no callback is added and nothing about the run differs.
    ap.add_argument('--ckpt-dir', default=None,
                    help='write the best-val_loss checkpoint and a true final-epoch '
                         'checkpoint here. Without it, only the end-of-fit weights are '
                         'saved, which are the restored best only when early stopping '
                         'fired -- not comparable across runs.')
    ap.add_argument('--run-tag', default=None,
                    help='filename prefix for those checkpoints (defaults to --tag)')
    ap.add_argument('--require-determinism', action='store_true',
                    help='hard-fail unless TF_DETERMINISTIC_OPS=1 and '
                         'TF_CUDNN_DETERMINISTIC=1 were exported by the launcher, before '
                         'this process started.')
    ap.add_argument('--lr', type=float, default=None,
                    help='constant learning rate, replacing the inherited train_one.py '
                         'schedule. That schedule opens at 1e-2 because it has to kick the '
                         'teacher\'s zero-init state correlator off its slow start; a plain '
                         'MLP/GRU has no such layer, so the opening LR may be too hot and '
                         'could destabilise the alpha=1 and alpha=0 arms asymmetrically. '
                         'Use this to pin an independent student LR before any sweep. '
                         'None => the teacher schedule (not recommended, unverified here).')
    ap.add_argument('--lr-schedule', choices=['constant', 'cosine'], default='constant',
                    help='with --lr: constant (default, as before) keeps --lr for every '
                         'update; cosine decays it per optimizer update from --lr to '
                         '--lr-end over all epochs x steps, no warmup '
                         '(helpers.cosine_lr_schedule).')
    ap.add_argument('--lr-warmup-epochs', type=float, default=0.0,
                    help='with --lr-schedule cosine: raise the rate linearly from ~0 to --lr '
                         'over this many epochs of updates, then run the cosine from --lr to '
                         '--lr-end over the remaining updates. Default 0 = no warmup, as before.')
    ap.add_argument('--lr-end', type=float, default=3e-5,
                    help='final learning rate of --lr-schedule cosine.')
    ap.add_argument('--ema-decay', type=float, default=None,
                    help='keep an exponential moving average of the weights (Adam use_ema, '
                         'ema_momentum = this value, e.g. 0.999). The val-LER checkpoint is '
                         'chosen on, and saves, the EMA weights; at the end of fit() Keras '
                         'copies the EMA weights into the model, so the end-of-fit score is '
                         'also the EMA weights. Needs --lr, --val-start and alpha 1. '
                         'Default None = no EMA, as before.')
    ap.add_argument('--global-clipnorm', type=float, default=None,
                    help='Adam global_clipnorm: when the global norm of all gradients of an '
                         'update exceeds this value, all of them are scaled down together '
                         'to it. Implies --log-grad-norm. Default None = no clipping, as '
                         'before.')
    ap.add_argument('--log-grad-norm', action='store_true',
                    help='record the global gradient norm (before clipping) of every '
                         'update; per epoch the median norm and the clipped fraction go '
                         'into the history, and all norms into <tag>.grad_norms.npy with a '
                         'summary in <tag>.grad_norms.json.')
    ap.add_argument('--stop-after-steps', type=int, default=None,
                    help='diagnostic: stop training after this many updates (the LR '
                         'schedule is still laid out over the full --epochs). Used to '
                         'measure gradient norms before choosing --global-clipnorm.')
    # Resume: continue a finished run from its last epoch to a larger --epochs. The model and
    # the Adam state (moments and update count) come from the run's .lastepoch.weights.h5;
    # the per-epoch shuffle is advanced past the epochs already done, so the resumed
    # epochs see the same data order an uninterrupted run would have.
    ap.add_argument('--resume-weights', default=None,
                    help='a .lastepoch.weights.h5 written by an earlier run (model + optimizer '
                         'state); training continues from it. Needs --initial-epoch.')
    ap.add_argument('--initial-epoch', type=int, default=0,
                    help='number of epochs already done by the run being resumed')
    ap.add_argument('--resume-best-json', default=None,
                    help='the earlier run\'s .best.json: the val-LER checkpoint starts from that '
                         'best, so a resumed epoch replaces it only if it is better. The '
                         'earlier .best.keras/.best.weights.h5 must already be in --ckpt-dir.')
    ap.add_argument('--resume-history', default=None,
                    help='the earlier run\'s .history.json; written history = earlier epochs + '
                         'resumed epochs')
    ap.add_argument('--warm-start', default=None,
                    help='a .best.keras from an earlier run (any distance). Every weight '
                         'tensor whose name and shape match this model\'s is copied in; the '
                         'others keep their fresh initialisation. Between distances the GRU '
                         'input kernel differs in shape (positions per timestep) and stays '
                         'fresh; the recurrent kernel, the GRU bias and the logit head load. '
                         'The optimizer starts fresh. Not with --resume-weights.')
    ap.add_argument('--max-params', type=int, default=None,
                    help='stop before training if the model has more parameters than this '
                         '(the width budget, e.g. the u280 count at this distance). Default '
                         'None = no limit.')
    ap.add_argument('--tail-diagnostics', action='store_true',
                    help='evaluate the tail every epoch and log p_L, predicted-positive '
                         'rate vs base rate, a class-collapse flag, and the student logit '
                         'mean/std into the history. Costs one tail inference per epoch.')
    # --- io ---
    ap.add_argument('--data-dir', default=os.path.expanduser('~/rcnn_threshold/pools'))
    ap.add_argument('--pool', default=None, help='explicit training pool npz')
    ap.add_argument('--extra-train-pool', nargs='*', default=[],
                    help='one or more pools, TRAINING ONLY, appended to the training prefix '
                         'in the order given. Their shots come from independent generation '
                         'seeds, so they cannot overlap the validation, evaluation or sealed '
                         'blocks of --pool. Use to train past a pool\'s clean prefix without '
                         'touching the reserved regions.')
    ap.add_argument('--extra-train-n', type=int, nargs='*', default=[],
                    help='how many shots to take from each --extra-train-pool, same order '
                         '(default: all of each)')
    ap.add_argument('--test-pool', default=None,
                    help='separate npz for a FRESH disjoint tail; strongly preferred')
    ap.add_argument('--eval-start', type=int, default=None,
                    help='first shot to score. This or --test-pool is required. With only '
                         '--test-pool, the last --n-test shots of that pool are scored. '
                         'Blocks overlapping the sealed block are refused.')
    ap.add_argument('--out-dir', default=None,
                    help='default: ~/rcnn_threshold/out_{student}, so MLP and GRU results '
                         'never land in the same directory')
    ap.add_argument('--tag', default=None, help='override the auto-generated output tag')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--gpu-mem-mib', type=int, default=None,
                    help='cap this process to N MiB of GPU memory, so several trainings '
                         'share one card without the first one parking all of it')
    args = ap.parse_args()

    # Name the architecture in every log line from here on: '[mlp]' or '[gru]'.
    global LOG
    LOG = f'[{args.student}]'
    if args.out_dir is None:
        args.out_dir = os.path.expanduser(f'~/rcnn_threshold/out_{args.student}')

    if args.teacher_model and args.teacher_cache:
        raise SystemExit(f"{LOG} give --teacher-model or --teacher-cache, not both. STOP.")
    if args.alpha < 1.0 and not (args.teacher_cache or args.teacher_model):
        raise SystemExit(f"{LOG} --teacher-cache or --teacher-model is required unless "
                         "--alpha 1.0 (pure hard-label control arm).")

    # --- guards that need no TensorFlow (2026-10-02) ----------------------------------
    # Extension shots get a teacher logit of 0 (p = 0.5), so with alpha < 1 the soft term
    # would train them toward a coin flip. No teacher cache exists for them.
    # (2026-10-06) With --teacher-model every shot, extension shots included, gets the
    # teacher's own logit, so the combination is allowed then.
    if args.extra_train_pool and args.alpha < 1.0 and not args.teacher_model:
        raise SystemExit(f"{LOG} --extra-train-pool with alpha={args.alpha} < 1: the extension "
                         f"shots have no teacher logits (they would be filled with 0, p=0.5). "
                         f"Use --alpha 1.0 with extensions. STOP.")
    # Without either flag the scored block would default to the last --n-test shots of the
    # primary pool, which on a FORMAL pool is the sealed block.
    if args.eval_start is None and args.test_pool is None:
        raise SystemExit(f"{LOG} give --eval-start (scored block in the primary pool) or "
                         f"--test-pool (a separate pool). STOP.")
    hidden = resolve_hidden(args.student, args.hidden)
    # One value: the int, so the model and every name built from it are as before.
    units = args.units[0] if len(args.units) == 1 else tuple(args.units)
    units_str = 'x'.join(str(u) for u in args.units)
    train_fn = args.pool or os.path.join(args.data_dir,
                                         f'data_d{args.d}_p{args.p:.3f}_r{args.rounds}.npz')
    if not os.path.exists(train_fn):
        raise SystemExit(f"{LOG} MISSING pool {train_fn}")
    check_pools_and_partitions(args, train_fn)

    if args.require_determinism:
        # The parameter-reduction study requires the LAUNCHER to have exported both
        # variables. The setdefault calls below would otherwise mask a launcher that
        # forgot, which is fine for this script alone but hides a broken launcher from
        # every other entry point in the same sweep.
        from helpers import assert_deterministic_env
        assert_deterministic_env()

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    # Same determinism flags as train_one_quantized.py. Without them, GPU float reductions
    # accumulate in a nondeterministic order and the same seed can hit a val_loss spike at
    # large data volume that the fixed LR decay never recovers from (diagnosed 2026-07-19).
    # Must be set before TensorFlow is imported so deterministic kernels are selected.
    os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
    os.environ.setdefault('TF_CUDNN_DETERMINISTIC', '1')
    import tensorflow as tf
    # Same hard version guard as the rest of the repo: TF is pinned at 2.15.x (Keras 2).
    # The student itself is Keras-3-safe, but the teacher cache and every sibling script
    # are not, and a mixed-version result would be silently incomparable.
    assert tf.__version__.startswith('2.15'), (
        f'need TF 2.15.x (Keras 2); got TF {tf.__version__}. Activate the pinned env.')
    if args.no_tf32:
        tf.config.experimental.enable_tensor_float_32_execution(False)
    print(f"{LOG} tf32={'off' if args.no_tf32 else 'on (TensorFlow default)'}", flush=True)
    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')
    elif args.gpu_mem_mib:
        cap_gpu_memory(tf, args.gpu_mem_mib)
    print(f"{LOG} TF {tf.__version__}  GPUs: {tf.config.list_physical_devices('GPU')}",
          flush=True)

    from types_cfg import get_types
    from circuit_partition import split_measurements
    from StudentModels import build_student, assemble_features, student_pred_and_correct

    d, p, r, k, seed = args.d, args.p, args.rounds, args.kernel, args.seed
    ntr, nte = args.n_train, args.n_test
    binary_t, _time_t, idx_t, _packed_t = get_types(d, r, k)

    ztr = np.load(train_fn)
    N = ztr['measurements'].shape[0]

    # --- training features ---------------------------------------------------------
    if args.test_pool:
        # Fresh disjoint tail: the training prefix may use the whole training pool.
        if ntr > N:
            raise SystemExit(f"{LOG} n_train={ntr} exceeds pool size {N}")
    else:
        # Same-pool split: the tail is the last nte shots, so the prefix must stop short.
        assert ntr <= N - nte, f"train/test overlap: ntr={ntr} > N-nte={N - nte}"

    # --- training-only extensions: checks and sizes first ----------------------------
    # (2026-10-01) The pools are checked and their sizes read before any features are
    # built, so the whole training set can be allocated once as uint8 and filled in place:
    # primary prefix first, then each extension in the order given (the same order and
    # values as before, when each pool was converted whole and concatenated). That removes
    # the full-size intermediate copies that set the host-memory peak.
    # Several extension pools are appended one after another, in the order given; one pool
    # behaves exactly as before. Each must have its own generation seed (checked against
    # the primary pool and against the other extensions) and no file may appear twice.
    ext_pools = list(args.extra_train_pool or [])
    ext_counts = list(args.extra_train_n or [])
    if ext_counts and len(ext_counts) != len(ext_pools):
        raise SystemExit(f"{LOG} --extra-train-n has {len(ext_counts)} values for "
                         f"{len(ext_pools)} --extra-train-pool files. STOP.")
    if len(set(map(os.path.abspath, ext_pools))) != len(ext_pools):
        raise SystemExit(f"{LOG} the same extension pool is listed twice. STOP.")
    ext_seeds = set()
    ext_plan = []                       # (path, take)
    for i_ex, ext_pool in enumerate(ext_pools):
        if not os.path.exists(ext_pool):
            raise SystemExit(f"{LOG} MISSING extra pool {ext_pool}")
        zex = np.load(ext_pool)
        n_ex = zex['flips'].shape[0]
        take = min((ext_counts[i_ex] if ext_counts else None) or n_ex, n_ex)
        fp = ext_pool.replace('.npz', '.fingerprint.json')
        if os.path.exists(fp):
            meta_ex = json.load(open(fp))
            if meta_ex.get('gen_seed') == _primary_gen_seed(train_fn):
                raise SystemExit(
                    f"{LOG} extra pool shares gen_seed {meta_ex.get('gen_seed')} with the "
                    f"primary pool. Same seed means the same shot stream -- the extension "
                    f"would duplicate training data and could repeat evaluation shots.")
            if meta_ex.get('gen_seed') in ext_seeds:
                raise SystemExit(
                    f"{LOG} two extension pools share gen_seed {meta_ex.get('gen_seed')}; "
                    f"their shots would be duplicates. STOP.")
            ext_seeds.add(meta_ex.get('gen_seed'))
            print(f"{LOG} extra pool gen_seed={meta_ex.get('gen_seed')} "
                  f"flips_sha={str(meta_ex.get('flips_sha256'))[:16]}", flush=True)
        ext_plan.append((ext_pool, take))
        del zex
    n_all = ntr + sum(t for _, t in ext_plan)

    # Primary training prefix, written into the start of the one training array.
    # Host-memory uint8 (see host_data.py): the features are 0/1, so this is exact.
    x_first, _ = load_extension_features(ztr, 1, d, r, p, args.inputs, args.student,
                                         idx_t, binary_t)          # shape of one shot
    x_tr = np.empty((n_all,) + x_first.shape[1:], dtype=np.uint8)
    del x_first
    _, f_tr = load_extension_features(ztr, ntr, d, r, p, args.inputs, args.student,
                                      idx_t, binary_t, out=x_tr[0:ntr])
    print(f"{LOG} training array {x_tr.shape} uint8 ({x_tr.nbytes / 1024 ** 3:.1f} GB) "
          f"allocated once; primary prefix [0, {ntr:,}) filled", flush=True)

    # --- targets: pack [hard_label, teacher_logit] ----------------------------------
    if args.teacher_model:
        # Filled after the training array is complete (below), from the teacher model.
        z_teach = np.zeros(ntr, dtype=np.float32)
        teacher_npz = None
    elif args.alpha >= 1.0 and not args.teacher_cache:
        # Hard-label control arm with no teacher available: the teacher column is unused
        # by the loss (weight 1-alpha == 0), so fill it with zeros rather than requiring
        # a cache the arm does not need.
        z_teach = np.zeros(ntr, dtype=np.float32)
        teacher_npz = None
        print(f"{LOG} hard-label control arm: teacher column unused (alpha=1).",
              flush=True)
    else:
        z_teach, p_teach, f_cache = load_teacher_cache(args.teacher_cache, train_fn, 0, ntr,
                                                       pool_npz=ztr)
        teacher_npz = np.load(args.teacher_cache, allow_pickle=False)
        # The cache carries its own copy of `flips`; if it disagrees with the pool's, the
        # two files are not describing the same shots and every soft target is misaligned.
        if not np.array_equal(f_cache.reshape(-1), f_tr.reshape(-1)):
            raise SystemExit(f"{LOG} teacher cache `flips` disagree with the pool's -- "
                             "the cache is misaligned with these shots.")
        print(f"{LOG} teacher cache ok: {ntr:,} shots, mean p_teacher={p_teach.mean():.5f}",
              flush=True)

    # --- optional independent training extension ------------------------------------
    # Appended to the training prefix only. The extension is a separate generation with its
    # own seed, so its shots are independent of every block of the primary pool; the
    # validation, evaluation and sealed regions are read from the primary pool alone and
    # are untouched by this.
    # The disjointness checks below are about index ranges inside the PRIMARY pool, so they
    # must keep testing the primary prefix even after the training set is augmented with
    # shots from a different pool.
    ntr_primary = ntr

    # Extensions, each written into its slice of the training array (checked above).
    for ext_pool, take in ext_plan:
        zex = np.load(ext_pool)
        _, f_ex = load_extension_features(zex, take, d, r, p, args.inputs, args.student,
                                          idx_t, binary_t, out=x_tr[ntr:ntr + take])
        f_tr = np.concatenate([f_tr, f_ex], axis=0)
        z_teach = np.concatenate([z_teach, np.zeros(take, dtype=np.float32)], axis=0)
        ntr = ntr + take
        del zex
        print(f"{LOG} training set extended by {take:,} shots from "
              f"{os.path.basename(ext_pool)}; total {ntr:,}", flush=True)
    assert ntr == n_all, (ntr, n_all)

    teacher = None
    if args.teacher_model:
        # Soft targets from the teacher model, computed on the training array itself (the
        # same uint8 features the student trains on, in the same order), so each shot's
        # target is the teacher's logit for that very shot.
        if not os.path.exists(args.teacher_model):
            raise SystemExit(f"{LOG} MISSING teacher model {args.teacher_model}")
        teacher = tf.keras.models.load_model(args.teacher_model, compile=False)
        if tuple(teacher.input_shape[1:]) != tuple(x_tr.shape[1:]):
            raise SystemExit(f"{LOG} teacher input {teacher.input_shape[1:]} != training "
                             f"features {x_tr.shape[1:]}. STOP.")
        t0_t = time.time()
        z_teach = teacher.predict(make_dataset(tf, x_tr, None, args.batch_size),
                                  verbose=0).reshape(-1).astype(np.float32)
        if z_teach.shape[0] != ntr:
            raise SystemExit(f"{LOG} teacher gave {z_teach.shape[0]} logits for {ntr} shots. STOP.")
        t_err = float(((z_teach > 0).astype(np.int8) != f_tr.reshape(-1).astype(np.int8)).mean())
        print(f"{LOG} teacher {os.path.basename(args.teacher_model)} "
              f"({teacher.count_params():,} params): logits for all {ntr:,} training shots in "
              f"{time.time() - t0_t:.0f}s; p_L on the training shots {t_err:.6f}; mean "
              f"sigmoid {float(np.mean(1 / (1 + np.exp(-z_teach.astype(np.float64))))):.5f}; "
              f"exactly-zero logits {int((z_teach == 0).sum())}", flush=True)

    prov = run_provenance(args, train_fn, teacher_npz)
    if args.teacher_model:
        prov['teacher_cache'] = 'model:' + os.path.basename(args.teacher_model)
        prov['teacher_weights_sha'] = sha16(args.teacher_model)
    print(f"{LOG} provenance: pool={prov['train_pool_dir']}/{prov['train_pool']} "
          f"flips_sha={prov['train_pool_flips_sha']} teacher_sha={prov['teacher_weights_sha']} "
          f"code_sha={prov['code_sha']}", flush=True)

    y_tr = np.stack([f_tr.astype(np.float32), z_teach], axis=1)

    # --- evaluation tail ------------------------------------------------------------
    zte = np.load(args.test_pool) if args.test_pool else ztr
    Nte = zte['measurements'].shape[0]
    if args.eval_start is None:
        te = slice(Nte - nte, Nte)
    else:
        if args.eval_start + nte > Nte:
            raise SystemExit(f"{LOG} --eval-start {args.eval_start} + n-test {nte} "
                             f"exceeds pool size {Nte}")
        te = slice(args.eval_start, args.eval_start + nte)
    print(f"{LOG} scoring shots [{te.start:,}, {te.stop:,})", flush=True)
    m_te = zte['measurements'][te].astype(binary_t)
    e_te = zte['det_evts'][te].astype(binary_t)
    f_te = zte['flips'][te].astype(binary_t).reshape(-1)
    b_te, _, _ = split_measurements(m_te, d, idx_t)
    x_te = to_host_uint8(assemble_features(b_te, e_te, args.inputs, student=args.student,
                                           d=d, rounds=r, p=p))
    del m_te, b_te, e_te

    # --- explicit validation block --------------------------------------------------
    # Mirrors train_one.py: when --val-start/--val-n are given, validation comes from
    # those shots and validation_split is disabled, so every shot in [0, n_train)
    # produces gradients. Disjointness is asserted rather than assumed.
    val_data = None
    if args.val_start is not None:
        if args.val_n is None:
            raise SystemExit(f"{LOG} --val-start requires --val-n")
        v0, v1 = args.val_start, args.val_start + args.val_n
        # Overlap with the training prefix, the scored block and the sealed block was
        # checked per file in check_pools_and_partitions; validation is read from the same
        # file as the scored block (zte), which is --test-pool when one is given.
        if v1 > Nte:
            raise SystemExit(f"{LOG} validation [{v0}, {v1}) exceeds pool size {Nte}")
        m_va = zte['measurements'][v0:v1].astype(binary_t)
        e_va = zte['det_evts'][v0:v1].astype(binary_t)
        f_va = zte['flips'][v0:v1].astype(binary_t).reshape(-1)
        b_va, _, _ = split_measurements(m_va, d, idx_t)
        x_va = to_host_uint8(assemble_features(b_va, e_va, args.inputs, student=args.student,
                                               d=d, rounds=r, p=p))
        del m_va, b_va, e_va
        # The loss expects the packed [hard_label, teacher_logit] target. With alpha=1 the
        # teacher column is unused, so zeros are honest filler. With alpha<1 they are not:
        # zeros are a teacher saying p=0.5 on every validation shot, which would make the
        # validation loss measure the wrong thing and, with early stopping, select on it.
        # So the logits must come from a cache covering the validation block.
        if args.alpha < 1.0 and teacher is not None:
            z_va = teacher.predict(make_dataset(tf, x_va, None, args.batch_size),
                                   verbose=0).reshape(-1).astype(np.float32)
            print(f"{LOG} teacher logits on the validation block: p_L "
                  f"{float(((z_va > 0).astype(np.int8) != f_va.reshape(-1).astype(np.int8)).mean()):.6f}",
                  flush=True)
        elif args.alpha < 1.0:
            if not args.val_teacher_cache:
                raise SystemExit(
                    f"{LOG} --val-start with alpha={args.alpha} < 1 requires "
                    f"--val-teacher-cache: the soft term needs teacher logits on the "
                    f"validation shots, and the training cache stops at n_train. Dump a "
                    f"cache over [{v0}, {v1}) and pass it (usually the same npz as "
                    f"--teacher-tail-cache).")
            z_va, _p_va, f_va_cache = load_teacher_cache(
                args.val_teacher_cache, train_fn, v0, v1, pool_npz=None)
            # The cache carries its own copy of the labels. If they disagree with the
            # pool's, the cache describes different shots than the ones just loaded --
            # which is exactly the silent-misalignment failure the fingerprints exist to
            # catch, so check it here too rather than trusting the range arithmetic.
            if not np.array_equal(f_va_cache.reshape(-1).astype(np.int8),
                                  f_va.reshape(-1).astype(np.int8)):
                raise SystemExit(
                    f"{LOG} --val-teacher-cache flips disagree with the pool over "
                    f"[{v0}, {v1}) -- the cache does not describe these shots.")
            z_va = z_va.reshape(-1).astype(np.float32)
        else:
            z_va = np.zeros(len(f_va), np.float32)
        y_va = np.stack([f_va.astype(np.float32), z_va], axis=1)
        val_data = (x_va, y_va)
        steps = int(np.ceil(ntr / args.batch_size))
        extra = ntr - ntr_primary
        print(f"{LOG} partitions: train [0, {ntr_primary:,}) of the primary pool"
              + (f" + {extra:,} augmentation shots" if extra else "")
              + f"  validation [{v0:,}, {v1:,})"
              + (f" of --test-pool {os.path.basename(args.test_pool)}" if args.test_pool else "")
              + "  (validation_split disabled)", flush=True)
        print(f"{LOG} {steps} steps/epoch x {args.epochs} epochs = "
              f"{steps * args.epochs:,} optimizer updates", flush=True)
    else:
        eff = int(ntr * (1 - args.val_split))
        steps = int(np.ceil(eff / args.batch_size))
        print(f"{LOG} WARNING: no --val-start; validation_split={args.val_split} carves "
              f"{ntr - eff:,} shots OUT of the training prefix, so only {eff:,} of "
              f"{ntr:,} produce gradients.", flush=True)
        print(f"{LOG} {steps} steps/epoch x {args.epochs} epochs = "
              f"{steps * args.epochs:,} optimizer updates", flush=True)

    # --- build, compile, fit --------------------------------------------------------
    set_seeds(seed)
    model = build_student(args.student, d=d, rounds=r, inputs=args.inputs,
                          hidden=hidden, units=units,
                          weight_bits=args.weight_bits, act_bits=args.act_bits,
                          reset_after=args.reset_after, unroll=args.unroll)
    print(f"{LOG} reset_after={args.reset_after} unroll={args.unroll}", flush=True)
    # The learning rate is a constant, or a per-update cosine decay over every update of
    # the run (steps per epoch x epochs). Gradient-norm logging and global clipping use a
    # subclass of Adam (helpers.make_adam_logging_grad_norm); without them the optimizer is
    # exactly the one built before.
    lr_value = None
    if args.lr is not None:
        warmup_steps = int(round(args.lr_warmup_epochs * steps))
        if warmup_steps and args.lr_schedule != 'cosine':
            raise SystemExit(f"{LOG} --lr-warmup-epochs needs --lr-schedule cosine. STOP.")
        lr_value = (warmup_cosine_lr_schedule(tf, args.lr, args.lr_end, steps * args.epochs,
                                              warmup_steps)
                    if args.lr_schedule == 'cosine' else args.lr)
        if warmup_steps:
            # The rate at the first updates, the end of the warmup and the last update, so
            # the log shows the schedule that was used.
            marks = sorted({0, 1, 2, warmup_steps - 1, warmup_steps, warmup_steps + 1,
                            steps * args.epochs - 1})
            print(f"{LOG} warmup {args.lr_warmup_epochs:g} epochs = {warmup_steps:,} updates; "
                  + "  ".join(f"lr[{m:,}]={float(lr_value(m)):.3e}" for m in marks), flush=True)
    elif args.lr_schedule != 'constant':
        raise SystemExit('--lr-schedule cosine needs --lr (the start rate)')
    elif args.lr_warmup_epochs:
        raise SystemExit(f"{LOG} --lr-warmup-epochs needs --lr. STOP.")
    log_gn = args.log_grad_norm or args.global_clipnorm is not None
    gn_stats = None
    adam_kw = {}
    if args.ema_decay is not None:
        if lr_value is None or val_data is None or args.alpha < 1.0:
            raise SystemExit(f"{LOG} --ema-decay needs --lr, --val-start/--val-n and alpha 1 "
                             f"(the EMA weights' validation loss is the hard-label loss). STOP.")
        adam_kw = dict(use_ema=True, ema_momentum=args.ema_decay)
    if log_gn:
        if lr_value is None:
            raise SystemExit('--log-grad-norm / --global-clipnorm need --lr')
        optimizer, gn_stats = make_adam_logging_grad_norm(
            tf, lr_value, args.global_clipnorm, steps * args.epochs, **adam_kw)
    else:
        optimizer = (tf.keras.optimizers.Adam(learning_rate=lr_value, **adam_kw)
                     if lr_value is not None else 'adam')
    print(f"{LOG} optimizer Adam lr={args.lr} schedule={args.lr_schedule}"
          + (f" lr_end={args.lr_end} over {steps * args.epochs:,} updates"
             if args.lr_schedule == 'cosine' else '')
          + (f" warmup_epochs={args.lr_warmup_epochs:g}" if args.lr_warmup_epochs else '')
          + (f" ema_decay={args.ema_decay:g}" if args.ema_decay is not None else '')
          + f" global_clipnorm={args.global_clipnorm} log_grad_norm={log_gn}", flush=True)
    model.compile(optimizer=optimizer,
                  loss=make_distillation_loss(args.alpha, args.temperature),
                  metrics=[make_hard_accuracy()])
    n_params = int(model.count_params())
    print(f"{LOG} {args.student} inputs={args.inputs} params={n_params:,}", flush=True)
    if args.max_params is not None and n_params > args.max_params:
        raise SystemExit(f"{LOG} {n_params:,} parameters exceed --max-params "
                         f"{args.max_params:,}. STOP.")
    if args.warm_start:
        if args.resume_weights:
            raise SystemExit(f"{LOG} --warm-start and --resume-weights together. STOP.")
        if not os.path.exists(args.warm_start):
            raise SystemExit(f"{LOG} MISSING warm-start model {args.warm_start}")
        # compile=False: only the architecture and weights are read, not the optimizer.
        src_model = tf.keras.models.load_model(args.warm_start, compile=False)
        src_w = {w.name: w for w in src_model.weights}
        n_loaded = 0
        for w in model.weights:
            s_w = src_w.get(w.name)
            if s_w is not None and tuple(s_w.shape) == tuple(w.shape):
                w.assign(s_w)
                n_loaded += 1
                print(f"{LOG} warm-start: loaded {w.name} {tuple(w.shape)}", flush=True)
            else:
                print(f"{LOG} warm-start: fresh  {w.name} {tuple(w.shape)} (source: "
                      f"{'absent' if s_w is None else tuple(s_w.shape)})", flush=True)
        if not n_loaded:
            raise SystemExit(f"{LOG} warm-start: no tensor of {args.warm_start} matches. STOP.")
        print(f"{LOG} warm-start from {args.warm_start}: {n_loaded}/{len(model.weights)} "
              f"tensors loaded", flush=True)
        del src_model, src_w
    if args.resume_weights:
        if args.initial_epoch <= 0:
            raise SystemExit(f"{LOG} --resume-weights needs --initial-epoch > 0")
        # The optimizer's variables must exist before its saved state can be loaded into
        # them; a .weights.h5 from model.save_weights holds both the model and the optimizer.
        model.optimizer.build(model.trainable_variables)
        model.load_weights(args.resume_weights)
        it = int(model.optimizer.iterations.numpy())
        print(f"{LOG} resumed from {args.resume_weights}: optimizer update count {it:,} "
              f"(expected {steps * args.initial_epoch:,} = {steps} x {args.initial_epoch})", flush=True)
        if it != steps * args.initial_epoch:
            raise SystemExit(f"{LOG} optimizer state not restored: update count {it} != "
                             f"{steps * args.initial_epoch}. STOP.")

    # Output tag, built before the checkpoint prefix so every file of the run shares it.
    # The pool directory is part of the auto tag: the same (student, alpha, seed, ntr)
    # trained on a different pool is a different experiment, and must not land on the same
    # filenames as an earlier one. Since 2026-10-02 the auto tag also names the width,
    # Dense layers, learning rate, schedule, epochs and reset_after, which used to differ
    # between runs that shared a tag. Launchers pass --tag, so their names are unchanged.
    wb = 'f32' if args.weight_bits is None else f'w{args.weight_bits}'
    ab = 'f32' if args.act_bits is None else f'a{args.act_bits}'
    pool_tag = os.path.basename(os.path.dirname(os.path.abspath(train_fn)))
    tag = args.tag or (f'student_{args.student}_{args.inputs.replace("+", "-")}_'
                       f'{wb}_{ab}_alpha{args.alpha:g}_T{args.temperature:g}_'
                       f'seed{seed}_ntr{ntr}_'
                       + (f'u{units_str}_' if args.student == 'gru' else '')
                       + f'h{"-".join(str(h) for h in hidden) or 0}_'
                       f'lr{"teacher" if args.lr is None else format(args.lr, "g")}_'
                       f'{args.lr_schedule}_e{args.epochs}_ra{int(args.reset_after)}_'
                       + (f'wu{args.lr_warmup_epochs:g}_' if args.lr_warmup_epochs else '')
                       + (f'ema{args.ema_decay:g}_' if args.ema_decay is not None else '')
                       + ('ws_' if args.warm_start else '')
                       + f'{pool_tag}')
    ck_name = args.run_tag or tag      # checkpoint and gradient-norm file prefix

    # With --lr the rate is constant and the teacher's scheduler is not attached; it would
    # overwrite the constant each epoch and turn an LR comparison into a schedule test.
    callbacks = []
    if args.lr_schedule == 'cosine':
        # Record the rate reached at the end of each epoch in the history (key 'lr').
        # Runs before Keras' History callback, so the value lands in history.json.
        def _log_lr(epoch, logs):
            logs['lr'] = float(lr_value(optimizer.iterations))
        callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=_log_lr))
    if gn_stats is not None:
        # Per epoch: median global gradient norm of the epoch's updates, and the fraction
        # of updates clipped in this epoch and since the start. The printed line is what
        # the queue driver watches ("clipped ... cumulative <fraction>").
        gn_prev = {'updates': 0, 'clipped': 0}

        def _log_gn(epoch, logs):
            n = int(gn_stats['n_updates'].numpy())
            c = int(gn_stats['n_clipped'].numpy())
            dn, dc = n - gn_prev['updates'], c - gn_prev['clipped']
            stored = gn_stats['norms'].numpy()
            norms = stored[gn_prev['updates']:min(n, len(stored))]
            logs['grad_norm_median'] = float(np.median(norms)) if len(norms) else float('nan')
            logs['clip_frac'] = dc / dn if dn else 0.0
            logs['clip_frac_cum'] = c / n if n else 0.0
            print(f"{LOG} epoch {epoch + 1}: global grad norm median {logs['grad_norm_median']:.4g}; "
                  f"clipped {dc:,}/{dn:,} updates ({logs['clip_frac']:.4f}), "
                  f"cumulative {logs['clip_frac_cum']:.4f}", flush=True)
            gn_prev['updates'], gn_prev['clipped'] = n, c
        callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=_log_gn))
    if args.stop_after_steps:
        def _stop(batch, logs):
            if int(optimizer.iterations.numpy()) >= args.stop_after_steps:
                model.stop_training = True
        callbacks.append(tf.keras.callbacks.LambdaCallback(on_train_batch_end=_stop))
    if args.lr is None:
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler))
    if args.tail_diagnostics:
        callbacks.append(TailDiagnostics(make_dataset(tf, x_te, None, args.batch_size),
                                         f_te, args.batch_size).build())
    if not args.no_early_stopping:
        callbacks.insert(0, tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=args.patience, restore_best_weights=True))
    if args.ckpt_dir:
        # Same construction train_one.py uses. The final-epoch saver is placed BEFORE
        # EarlyStopping, because EarlyStopping(restore_best_weights=True) swaps the best
        # weights back in during its own on_epoch_end on the stopping epoch, and anything
        # that must observe the true final epoch has to run first.
        os.makedirs(args.ckpt_dir, exist_ok=True)
        _ck = os.path.join(args.ckpt_dir, ck_name)

        class _SaveLastEpoch(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                self.model.save_weights(_ck + '.lastepoch.weights.h5')

        callbacks.insert(0, _SaveLastEpoch())
        if val_data is not None:
            # Checkpoint rule: keep the epoch with the lowest LER on the explicit
            # validation block (tiebreak: lower val_loss), saved as a full .keras model
            # plus .weights.h5. Inserted first so it scores the weights this epoch
            # actually ended with, before EarlyStopping can restore older ones.
            from helpers import make_val_ler_checkpoint
            callbacks.insert(0, make_val_ler_checkpoint(
                tf, make_dataset(tf, x_va, None, args.batch_size), f_va, _ck,
                args.batch_size, LOG, previous_best=(json.load(open(args.resume_best_json))
                                                     if args.resume_best_json else None),
                ema_optimizer=(optimizer if args.ema_decay is not None else None)))
            print(f"{LOG} checkpoints -> {_ck}.best.keras + .best.weights.h5 "
                  f"(lowest val LER, tiebreak val_loss; epoch in {_ck}.best.json) and "
                  f"{_ck}.lastepoch.weights.h5", flush=True)
        else:
            # No explicit validation block (validation_split mode): there are no
            # validation labels to score LER on, so the old val_loss selection is kept.
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                _ck + '.best.weights.h5', monitor='val_loss', mode='min',
                save_best_only=True, save_weights_only=True, verbose=0))
            print(f"{LOG} WARNING: no --val-start block, so the checkpoint is chosen by "
                  f"min val_loss, not val LER -> {_ck}.best.weights.h5", flush=True)
    # Printed, not asserted: EarlyStopping(restore_best_weights=True) swaps weights during
    # its own on_epoch_end, so anything that must observe the true final epoch has to
    # appear before it. Logging the constructed order makes that checkable by inspection.
    print(f"{LOG} callback order: "
          f"{[type(c).__name__ for c in callbacks]}", flush=True)

    t0 = time.time()
    if val_data is not None:
        # Host-memory pipelines (host_data.py): training batches are drawn from a seeded
        # per-epoch shuffle of all training indices; validation in shot order. The shuffle
        # order differs from Keras' numpy shuffle, so runs are not bit-for-bit comparable
        # with runs from the old pipeline.
        train_ds = make_dataset(tf, x_tr, y_tr, args.batch_size, shuffle_seed=seed)
        if args.initial_epoch:
            # Keras opens one iterator of train_ds per epoch, and each new iterator draws
            # the next shuffle order. Opening (and closing) one iterator per epoch already
            # done moves the order forward to where an uninterrupted run would be.
            for _ in range(args.initial_epoch):
                _it = iter(train_ds)
                del _it
            print(f"{LOG} shuffle order advanced past {args.initial_epoch} epochs", flush=True)
        val_ds = make_dataset(tf, x_va, y_va, args.batch_size)
        print(f"{LOG} input: host-memory uint8 tf.data, shuffle seed {seed}, "
              f"batch {args.batch_size}, prefetch to GPU", flush=True)
        hist = model.fit(train_ds, epochs=args.epochs, verbose=2, callbacks=callbacks,
                         validation_data=val_ds, initial_epoch=args.initial_epoch)
    else:
        # validation_split mode (no explicit val block) cannot split a tf.data pipeline;
        # it keeps the old numpy path. Not used by the launchers.
        hist = model.fit(x=x_tr.astype(np.float32), y=y_tr, batch_size=args.batch_size,
                         epochs=args.epochs, shuffle=True, verbose=2, callbacks=callbacks,
                         validation_split=args.val_split)
    train_time = time.time() - t0
    epochs_ran = len(hist.history['loss'])
    if gn_stats is not None and args.out_dir:
        # Every update's global gradient norm (before clipping), and a summary.
        n = int(gn_stats['n_updates'].numpy())
        norms = gn_stats['norms'].numpy()[:n]
        base = os.path.join(args.out_dir, ck_name + '.grad_norms')
        os.makedirs(args.out_dir, exist_ok=True)
        np.save(base + '.npy', norms)
        q = [0, 1, 5, 25, 50, 75, 95, 99, 100]
        with open(base + '.json', 'w') as fh:
            json.dump(dict(n_updates=n, n_clipped=int(gn_stats['n_clipped'].numpy()),
                           global_clipnorm=args.global_clipnorm,
                           percentiles={str(k): float(v) for k, v in
                                        zip(q, np.percentile(norms, q))} if n else {}),
                      fh, indent=1)
        print(f"{LOG} gradient norms of {n:,} updates -> {base}.npy/.json", flush=True)
    best_val = float(min(hist.history['val_loss']))

    # --- score on the tail ----------------------------------------------------------
    logits = model.predict(make_dataset(tf, x_te, None, args.batch_size), verbose=0)
    pred, correct = student_pred_and_correct(logits, f_te)
    pL = float((~correct).mean())
    base_rate = float(f_te.mean())
    beats_base = pL < base_rate

    # MWPM baselines are tail-specific. lookup_mwpm returns the one computed on the
    # --data-dir pool's own tail, so under an explicit --test-pool it describes different
    # shots and no ratio from it is meaningful. Decode MWPM on the actual tail
    # (eval_on_tail.py --mcnemar) instead.
    if args.test_pool or args.eval_start is not None:
        mwpm = None
        given = ' and '.join(f for f, v in (('--test-pool', args.test_pool),
                                             ('--eval-start', args.eval_start)) if v is not None)
        print(f"{LOG} {given} given: suppressing the stored MWPM ratio (baselines "
              "are tail-specific). Decode MWPM on THIS block to compare.", flush=True)
    else:
        mwpm = lookup_mwpm(args.data_dir, d, p, r)

    # --- distillation gap: how closely does the student track the teacher? ----------
    # Measured on decisions, not probabilities: a student can differ in probability
    # everywhere and still decode identically. Needs a tail teacher cache.
    ag = {k: '' for k in ('agree_all', 'agree_ambiguous', 'agree_confident', 'n_ambiguous',
                          'frac_ambiguous', 'student_p_L_ambiguous', 'teacher_p_L_ambiguous',
                          'teacher_tail_p_L')}
    if teacher is not None:
        # Teacher on the scored block, for the student-vs-teacher agreement columns.
        z_te = teacher.predict(make_dataset(tf, x_te, None, args.batch_size),
                               verbose=0).reshape(-1).astype(np.float64)
        ag = teacher_agreement(pred, 1 / (1 + np.exp(-z_te)), f_te, tuple(args.ambiguous_band))
        print(f"{LOG} agreement vs teacher on the scored block: all={ag['agree_all']}  "
              f"ambiguous={ag['agree_ambiguous']}  teacher p_L={ag['teacher_tail_p_L']}", flush=True)
    elif args.teacher_tail_cache:
        if not os.path.exists(args.teacher_tail_cache):
            raise SystemExit(f"{LOG} MISSING tail cache {args.teacher_tail_cache}")
        tc = np.load(args.teacher_tail_cache)
        if tc['p_teacher'].shape[0] == nte:
            ag = teacher_agreement(pred, tc['p_teacher'], f_te, tuple(args.ambiguous_band))
            print(f"{LOG} agreement vs teacher: all={ag['agree_all']}  "
                  f"ambiguous={ag['agree_ambiguous']} (band {args.ambiguous_band[0]}-"
                  f"{args.ambiguous_band[1]}, n={ag['n_ambiguous']:,} = "
                  f"{ag['frac_ambiguous']:.1%})  confident={ag['agree_confident']}", flush=True)
            print(f"{LOG}   inside the band: student p_L={ag['student_p_L_ambiguous']}  "
                  f"teacher p_L={ag['teacher_p_L_ambiguous']}", flush=True)
        else:
            print(f"{LOG} tail teacher cache has {tc['p_teacher'].shape[0]} shots, "
                  f"tail is {nte} -- skipping agreement.", flush=True)

    # --- write results --------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    # (tag: built above, before the checkpoint prefix)

    # Never silently replace an existing result. Collisions are archived with the run's
    # timestamp rather than overwritten, so an accidental re-run cannot destroy a number
    # that has not been written up yet.
    csv_path = os.path.join(args.out_dir, tag + '.csv')
    if os.path.exists(csv_path):
        stamp = prov['run_utc'].replace(':', '').replace('-', '')
        for ext in ('.csv', '.history.json', '.weights.h5'):
            old = os.path.join(args.out_dir, tag + ext)
            if os.path.exists(old):
                os.rename(old, os.path.join(args.out_dir, f'{tag}.superseded_{stamp}{ext}'))
        print(f"{LOG} existing results for this tag archived as "
              f"{tag}.superseded_{stamp}.*", flush=True)

    fields = ['architecture', 'student', 'inputs', 'd', 'p', 'rounds', 'seed', 'n_train',
              'n_test', 'alpha', 'temperature', 'lr', 'lr_schedule', 'lr_end', 'lr_warmup_epochs',
              'ema_decay', 'warm_start', 'global_clipnorm',
              'clip_frac',
              'weight_bits', 'act_bits',
              # architecture size, so a row is interpretable without parsing its tag
              'hidden', 'units', 'n_params', 'reset_after', 'unroll', 'tf32',
              'epochs', 'epochs_ran', 'batch_size', 'p_L', 'scored_weights',
              'mwpm_p_L', 'ratio_vs_mwpm',
              'base_rate', 'beats_base_rate', 'teacher_tail_p_L',
              # agreement: the raw rate is dominated by the ~80% of shots where this
              # teacher is confident, so agree_ambiguous is the one that carries signal
              'agree_all', 'agree_ambiguous', 'agree_confident', 'n_ambiguous',
              'frac_ambiguous', 'student_p_L_ambiguous', 'teacher_p_L_ambiguous',
              'ambiguous_band', 'best_val_loss', 'train_time_s'] + PROVENANCE_COLS
    with open(os.path.join(args.out_dir, tag + '.csv'), 'w', newline='') as cf:
        w = csv.DictWriter(cf, fieldnames=fields)
        w.writeheader()
        w.writerow(dict(
            architecture=model.name, student=args.student, inputs=args.inputs,
            d=d, p=p, rounds=r, seed=seed, n_train=ntr, n_test=nte,
            alpha=args.alpha, temperature=args.temperature,
            lr=('teacher_schedule' if args.lr is None else args.lr),
            lr_schedule=args.lr_schedule,
            lr_end=(args.lr_end if args.lr_schedule == 'cosine' else ''),
            lr_warmup_epochs=args.lr_warmup_epochs,
            ema_decay=('' if args.ema_decay is None else args.ema_decay),
            warm_start=(os.path.abspath(args.warm_start) if args.warm_start else ''),
            global_clipnorm=('' if args.global_clipnorm is None else args.global_clipnorm),
            clip_frac=('' if gn_stats is None or not int(gn_stats['n_updates'].numpy())
                       else round(int(gn_stats['n_clipped'].numpy())
                                  / int(gn_stats['n_updates'].numpy()), 6)),
            weight_bits=('' if args.weight_bits is None else args.weight_bits),
            act_bits=('' if args.act_bits is None else args.act_bits),
            hidden='-'.join(str(h) for h in hidden),
            units=(units_str if args.student == 'gru' else ''),
            n_params=n_params, reset_after=int(args.reset_after), unroll=int(args.unroll),
            tf32=('off' if args.no_tf32 else 'on'),
            epochs=args.epochs, epochs_ran=epochs_ran,
            batch_size=args.batch_size, p_L=round(pL, 6),
            # p_L above is from the weights the model holds when fit() returns: the last
            # epoch's, or with early stopping on, possibly the restored min-val_loss epoch.
            # The best-checkpoint number of record comes from eval_student_on_tail.py.
            scored_weights=(('end_of_fit' if args.no_early_stopping
                             else 'end_of_fit_early_stopping_restore')
                            + ('_ema' if args.ema_decay is not None else '')),
            mwpm_p_L=('' if mwpm is None else round(mwpm, 6)),
            ratio_vs_mwpm=('' if mwpm is None else round(pL / mwpm, 4)),
            base_rate=round(base_rate, 5), beats_base_rate=int(beats_base),
            ambiguous_band=f'{args.ambiguous_band[0]}-{args.ambiguous_band[1]}',
            best_val_loss=round(best_val, 5), train_time_s=round(train_time, 1),
            **ag, **prov))
    hist_out = {k2: [float(x) for x in v] for k2, v in hist.history.items()}
    if args.resume_history:
        # Earlier epochs first, then the resumed ones, so the file covers the whole run.
        prev = json.load(open(args.resume_history))
        if len(prev['val_loss']) != args.initial_epoch:
            raise SystemExit(f"{LOG} --resume-history has {len(prev['val_loss'])} epochs, "
                             f"--initial-epoch is {args.initial_epoch}. STOP.")
        hist_out = {k2: list(prev.get(k2, [])) + v for k2, v in hist_out.items()}
    with open(os.path.join(args.out_dir, tag + '.history.json'), 'w') as hf:
        json.dump(hist_out, hf)
    wpath = os.path.join(args.out_dir, tag + '.weights.h5')
    model.save_weights(wpath)

    gap = '' if mwpm is None else (f'  MWPM(same-pool tail)={mwpm:.5f}  '
                                   f'ratio={pL / mwpm:.3f}x')
    flag = '' if beats_base else '  <-- fails: p_L >= base rate (class collapse)'
    print(f"{LOG} {tag}  p_L={pL:.5f}{gap}  base_rate={base_rate:.3f}"
          f"  agree_ambiguous={ag['agree_ambiguous']}{flag}"
          f"  ({epochs_ran} ep, {train_time:.0f}s)",
          flush=True)
    print(f"{LOG} wrote -> {os.path.join(args.out_dir, tag + '.csv')}  (+ history, weights)",
          flush=True)


if __name__ == '__main__':
    run()
