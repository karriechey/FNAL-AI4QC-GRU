# Created: 2026-09-25
# Last modified: 2026-09-27 (cosine_lr_schedule, make_adam_logging_grad_norm)
"""Small shared helpers for the GRU pipeline.

The first four functions are moved verbatim from the old repo's RCNN files so that no RCNN
code has to come across with them:

  set_seeds, learning_rate_scheduler (+ LR_FLOOR), lookup_mwpm
      from FNAL-QCDecoding-FPGA/train_one.py @ d5ee86f
  mcnemar_from_correct
      from FNAL-QCDecoding-FPGA/eval_on_tail.py @ d5ee86f
  assert_deterministic_env (+ REQUIRED_ENV)
      from FNAL-QCDecoding-FPGA/slice_guard_r5.py @ d5ee86f (the rest of that file,
      including its separate r5 partition layout, is not carried over)

Their bodies are unchanged; tests/ proves they give the same results as the originals.
(eval_on_tail.py has its own copy of lookup_mwpm with an identical body; the train_one.py
copy is the one train_student.py used, so that is the one kept here.)

make_val_ler_checkpoint is new. It implements the checkpoint rule for every training run:
score the validation block once per epoch, log val LER next to val loss, and keep the
epoch with the lowest val LER (fewest logical errors), breaking ties by the lower val loss.
The kept model is saved both as a full .keras file (graph + weights, loads without the
model-building code) and as .weights.h5 (what the existing eval scripts load).
"""
import csv
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------------------
# from FNAL-QCDecoding-FPGA/train_one.py @ d5ee86f
# ---------------------------------------------------------------------------------------
LR_FLOOR = 0.0


def set_seeds(seed):
    import random
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def learning_rate_scheduler(epoch, lr):
    """the reference architecture's schedule (surface_code_d5_r3_RCNN.ipynb): high at epoch 0 to move the
    slow-start zero-init state correlator, then decay."""
    if epoch < 10:
        sched = 0.001 * (10 - epoch)
    elif epoch < 20:
        sched = lr * 0.9
    elif epoch < 30:
        sched = lr * 0.8
    else:
        sched = lr * 0.65
    return max(sched, LR_FLOOR)


def cosine_lr_schedule(tf, lr_start, lr_end, total_steps):
    """Cosine decay of the learning rate from lr_start to lr_end over total_steps optimizer
    updates, no warmup, updated every step (not every epoch):

        lr(step) = lr_end + (lr_start - lr_end) * 0.5 * (1 + cos(pi * step / total_steps))

    Keras' CosineDecay computes exactly this when alpha = lr_end / lr_start (its alpha is
    the final rate as a fraction of the initial one). Steps past total_steps stay at lr_end.
    Added for the 50M runs, whose training loss rose late in training at a constant 3e-3.
    """
    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr_start, decay_steps=int(total_steps),
        alpha=lr_end / lr_start)


def make_adam_logging_grad_norm(tf, learning_rate, global_clipnorm, n_steps_total):
    """Adam that records the global gradient norm of every update, before clipping.

    The global norm is sqrt(sum over all weight tensors of sum(g**2)), the quantity Keras'
    global_clipnorm compares with its threshold: if it exceeds global_clipnorm, every
    gradient is scaled by global_clipnorm / norm (tf.clip_by_global_norm). With
    global_clipnorm=None nothing is clipped and the norms are only recorded.

    Returns (optimizer, stats). stats holds tf.Variables, kept outside the optimizer so they
    are never saved with it:
      norms      float32 (n_steps_total,)  norm of update i at index i (updates past
                                          n_steps_total are counted but not stored)
      n_updates  int64  updates so far
      n_clipped  int64  updates whose norm exceeded global_clipnorm
    """
    stats = dict(norms=tf.Variable(tf.zeros([int(n_steps_total)], tf.float32), trainable=False),
                 n_updates=tf.Variable(0, dtype=tf.int64, trainable=False),
                 n_clipped=tf.Variable(0, dtype=tf.int64, trainable=False))
    size = int(n_steps_total)

    class AdamLoggingGradNorm(tf.keras.optimizers.Adam):
        # Keras 2.15 calls _clip_gradients(grads) once per update inside apply_gradients,
        # before the Adam step; the norm is taken here on the unclipped gradients.
        def _clip_gradients(self, grads):
            norm = tf.linalg.global_norm([g for g in grads if g is not None])
            i = tf.cast(stats['n_updates'], tf.int32)
            tf.cond(i < size,
                    lambda: stats['norms'].scatter_nd_update([[i]], [norm]),
                    lambda: stats['norms'])
            stats['n_updates'].assign_add(1)
            if global_clipnorm:
                stats['n_clipped'].assign_add(tf.cast(norm > global_clipnorm, tf.int64))
            return super()._clip_gradients(grads)

    kw = {} if global_clipnorm is None else {'global_clipnorm': global_clipnorm}
    return AdamLoggingGradNorm(learning_rate=learning_rate, **kw), stats


def lookup_mwpm(data_dir, d, p, rounds):
    """Read the canonical once-per-(d,p,rounds) MWPM p_L, so the result row carries the
    target the RCNN is chasing. Returns None if unavailable.

    NOTE: matches on rounds too. The same (d,p) can have multiple pools at different
    rounds (e.g. d5/p0.010 exists at BOTH r5=0.0891 and r3~0.048); matching on (d,p)
    alone silently attaches the wrong-rounds MWPM to a run."""
    path = os.path.join(data_dir, "mwpm_baseline.csv")
    if not os.path.exists(path):
        return None
    for row in csv.DictReader(open(path)):
        if (int(row["d"]) == d and abs(float(row["p"]) - p) < 1e-9
                and int(row["rounds"]) == rounds):
            return float(row["mwpm_p_L"])
    return None


# ---------------------------------------------------------------------------------------
# from FNAL-QCDecoding-FPGA/eval_on_tail.py @ d5ee86f
# ---------------------------------------------------------------------------------------
def mcnemar_from_correct(rcnn_correct, mwpm_correct):
    """Paired 2x2 + McNemar from per-shot boolean correctness arrays.
    Returns dict with the four raw counts and the test stats. b = RCNN-only-correct
    (RCNN wins the shot), c = MWPM-only-correct (MWPM wins)."""
    rc = np.asarray(rcnn_correct, dtype=bool)
    mc = np.asarray(mwpm_correct, dtype=bool)
    both_right = int((rc & mc).sum())
    rcnn_only = int((rc & ~mc).sum())    # b: RCNN right, MWPM wrong -> RCNN wins
    mwpm_only = int((~rc & mc).sum())    # c: MWPM right, RCNN wrong -> MWPM wins
    both_wrong = int((~rc & ~mc).sum())
    b, c = rcnn_only, mwpm_only
    n_disc = b + c
    # McNemar with continuity correction -> chi-square, 1 dof
    if n_disc > 0:
        chi2_cc = (abs(b - c) - 1) ** 2 / n_disc
    else:
        chi2_cc = 0.0
    # p-values: chi2 approx (safe, no version issues) + exact binomial if scipy has it
    try:
        from scipy.stats import chi2 as _chi2
        p_chi2 = float(_chi2.sf(chi2_cc, 1))
    except Exception:
        p_chi2 = float('nan')
    p_exact = float('nan')
    try:
        from scipy.stats import binomtest
        p_exact = float(binomtest(b, n_disc, 0.5, alternative='two-sided').pvalue) if n_disc else 1.0
    except Exception:
        try:
            from scipy.stats import binom_test
            p_exact = float(binom_test(b, n_disc, 0.5)) if n_disc else 1.0
        except Exception:
            pass
    return dict(both_right=both_right, rcnn_only=rcnn_only, mwpm_only=mwpm_only,
                both_wrong=both_wrong, n_discordant=n_disc, net_rcnn_wins=b - c,
                mcnemar_chi2_cc=chi2_cc, p_chi2=p_chi2, p_exact=p_exact)


# ---------------------------------------------------------------------------------------
# from FNAL-QCDecoding-FPGA/slice_guard_r5.py:192-217 @ d5ee86f
# ---------------------------------------------------------------------------------------
REQUIRED_ENV = {'TF_DETERMINISTIC_OPS': '1', 'TF_CUDNN_DETERMINISTIC': '1'}


def assert_deterministic_env():
    """Fail unless the launcher exported the determinism variables BEFORE this process.

    Must be called before `import tensorflow`. TensorFlow reads both variables during its
    own import, so a script that sets them afterwards gets no determinism and no warning.
    This function refuses to set them itself for exactly that reason: if they are missing,
    the process was started wrong and cannot be repaired in place.
    """
    if 'tensorflow' in sys.modules:
        raise AssertionError(
            "assert_deterministic_env() was called after TensorFlow was already imported. "
            "The determinism variables are read at import time, so this ordering gives no "
            "determinism. Move the call above the import.")
    missing = [k for k, v in REQUIRED_ENV.items() if os.environ.get(k) != v]
    if missing:
        raise SystemExit(
            "[env] MISSING required determinism variables: "
            + ', '.join(f"{k}={REQUIRED_ENV[k]}" for k in missing)
            + ". Export them in the launcher, before the Python process starts:\n"
              "    export TF_DETERMINISTIC_OPS=1\n"
              "    export TF_CUDNN_DETERMINISTIC=1")
    print("[env] TF_DETERMINISTIC_OPS=1 TF_CUDNN_DETERMINISTIC=1 present before import",
          flush=True)
    return True


# ---------------------------------------------------------------------------------------
# New: checkpoint selection by validation LER
# ---------------------------------------------------------------------------------------
def make_val_ler_checkpoint(tf, x_val, flips_val, ckpt_prefix, batch_size, log_prefix=''):
    """Build a Keras callback that selects the checkpoint by validation LER.

    Every epoch it predicts the whole validation block, counts logical errors with the
    student decision rule (logit > 0 means "flip", the same threshold as
    StudentModels.student_pred_and_correct), and adds `val_ler` and `val_errors` to the
    epoch logs, so they land in the Keras history next to `val_loss`.

    Selection: fewest validation errors wins (errors are integers, so there is no
    floating-point tie ambiguity); on an exact tie the lower val_loss wins. When an epoch
    becomes the new best it writes

        <ckpt_prefix>.best.keras        full model (graph + weights)
        <ckpt_prefix>.best.weights.h5   weights only, for the existing eval scripts
        <ckpt_prefix>.best.json         chosen epoch, val LER, error count, val size, val loss

    The callback must be given the validation block only. It never sees the evaluation
    block or the sealed partition; the caller chooses the arrays.
    """
    flips_val = np.asarray(flips_val).reshape(-1).astype(np.int8)
    n_val = int(flips_val.shape[0])

    class ValLERCheckpoint(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best_errors = None
            self.best_val_loss = None
            self.best_epoch = None      # 1-based, to match the printed epoch numbers

        def on_epoch_end(self, epoch, logs=None):
            logs = logs if logs is not None else {}
            kw = {} if hasattr(x_val, 'element_spec') else dict(batch_size=batch_size)
            logits = self.model.predict(x_val, verbose=0, **kw)
            pred = (np.asarray(logits).reshape(-1) > 0.0).astype(np.int8)
            n_err = int((pred != flips_val).sum())
            ler = n_err / n_val
            val_loss = float(logs.get('val_loss', float('nan')))
            # Written into logs so Keras' History (which runs after user callbacks)
            # records them per epoch alongside loss and val_loss.
            logs['val_ler'] = ler
            logs['val_errors'] = n_err

            better = (self.best_errors is None or n_err < self.best_errors
                      or (n_err == self.best_errors and val_loss < self.best_val_loss))
            if better:
                self.best_errors, self.best_val_loss = n_err, val_loss
                self.best_epoch = epoch + 1
                self.model.save(ckpt_prefix + '.best.keras')
                self.model.save_weights(ckpt_prefix + '.best.weights.h5')
                with open(ckpt_prefix + '.best.json', 'w') as fh:
                    json.dump(dict(best_epoch=self.best_epoch, val_ler=ler,
                                   val_errors=n_err, val_n=n_val, val_loss=val_loss,
                                   selection='min val_errors, tiebreak min val_loss'),
                              fh, indent=1)
            print(f"{log_prefix} epoch {epoch + 1}: val_loss={val_loss:.5f} "
                  f"val_ler={ler:.6f} ({n_err:,}/{n_val:,} errors)"
                  f"{'  <- new best' if better else ''}", flush=True)

    return ValLERCheckpoint()
