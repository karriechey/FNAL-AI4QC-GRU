# Created: 2026-09-25
# Last modified: 2026-10-03 (warmup_cosine_lr_schedule: linear warmup before the cosine; EMA: make_adam_logging_grad_norm passes Adam options through, make_val_ler_checkpoint scores and saves the EMA weights when given the optimizer).
# Earlier 2026-09-29 (make_val_ler_checkpoint: previous_best for resumed runs).
# Earlier 2026-09-27 (cosine_lr_schedule, make_adam_logging_grad_norm)

"""Shared helpers for the GRU pipeline.

set_seeds, learning_rate_scheduler, lookup_mwpm, mcnemar_from_correct and
assert_deterministic_env are copied unchanged from FNAL-QCDecoding-FPGA @ d5ee86f
(source file noted above each); tests/ checks they give the same results.
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
    # reference schedule (surface_code_d5_r3_RCNN.ipynb): high at epoch 0, then decay
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
    # Per-step cosine decay lr_start -> lr_end over total_steps, no warmup.
    # Keras CosineDecay with alpha = lr_end / lr_start; stays at lr_end after total_steps.
    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr_start, decay_steps=int(total_steps),
        alpha=lr_end / lr_start)


def warmup_cosine_lr_schedule(tf, lr_start, lr_end, total_steps, warmup_steps):
    # Linear warmup over warmup_steps, then cosine_lr_schedule over the remaining steps.
    # warmup_steps = 0 returns cosine_lr_schedule unchanged.
    if not warmup_steps:
        return cosine_lr_schedule(tf, lr_start, lr_end, total_steps)
    w = int(warmup_steps)
    if w >= int(total_steps):
        raise ValueError(f'warmup {w} updates >= total {int(total_steps)} updates')
    cosine = cosine_lr_schedule(tf, lr_start, lr_end, int(total_steps) - w)

    @tf.keras.saving.register_keras_serializable(package='gru_pipeline')
    class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            warm = lr_start * (step + 1.0) / float(w)
            return tf.where(step < float(w), warm, cosine(tf.maximum(step - float(w), 0.0)))

        def get_config(self):
            return dict(lr_start=lr_start, lr_end=lr_end, total_steps=int(total_steps),
                        warmup_steps=w)

    return WarmupCosine()


def make_adam_logging_grad_norm(tf, learning_rate, global_clipnorm, n_steps_total, **adam_kw):
    # Adam that records the global gradient norm of every update, before clipping.
    # Returns (optimizer, stats); stats = norms (n_steps_total,), n_updates, n_clipped,
    # kept outside the optimizer so they are not saved with it.
    # adam_kw: extra Adam arguments, e.g. use_ema=True, ema_momentum=0.999.
    stats = dict(norms=tf.Variable(tf.zeros([int(n_steps_total)], tf.float32), trainable=False),
                 n_updates=tf.Variable(0, dtype=tf.int64, trainable=False),
                 n_clipped=tf.Variable(0, dtype=tf.int64, trainable=False))
    size = int(n_steps_total)

    class AdamLoggingGradNorm(tf.keras.optimizers.Adam):
        # Keras 2.15 calls this once per update, before the Adam step, on unclipped gradients
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
    return AdamLoggingGradNorm(learning_rate=learning_rate, **kw, **adam_kw), stats


def lookup_mwpm(data_dir, d, p, rounds):
    # MWPM p_L for (d, p, rounds), or None. Matches on rounds too: the same (d, p) can
    # have pools at different rounds.
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
    # Paired 2x2 + McNemar from per-shot correctness.
    # b = RCNN right / MWPM wrong, c = MWPM right / RCNN wrong.
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
    # The launcher must export the determinism variables: TensorFlow reads them during its
    # own import, so setting them here would be too late. Call before `import tensorflow`.
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
# checkpoint selection by validation LER
# ---------------------------------------------------------------------------------------
def make_val_ler_checkpoint(tf, x_val, flips_val, ckpt_prefix, batch_size, log_prefix='',
                            previous_best=None, ema_optimizer=None):
    # Keras callback: each epoch, score the validation block (logit > 0 = flip), add val_ler
    # and val_errors to the logs, and keep the epoch with the fewest errors (tiebreak val_loss).
    # Saves <ckpt_prefix>.best.keras, .best.weights.h5, .best.json.
    # previous_best: .best.json of a resumed run, so selection continues from it.
    # ema_optimizer: Adam with use_ema=True; the EMA weights are scored and saved, and their
    # loss (hard-label BCE, alpha = 1 only) is used as val_loss_ema for the tiebreak.
    flips_val = np.asarray(flips_val).reshape(-1).astype(np.int8)
    n_val = int(flips_val.shape[0])

    class ValLERCheckpoint(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best_errors = None
            self.best_val_loss = None
            self.best_epoch = None      # 1-based, to match the printed epoch numbers
            if previous_best is not None:
                if int(previous_best['val_n']) != n_val:
                    raise ValueError('previous_best was scored on a different validation size')
                self.best_errors = int(previous_best['val_errors'])
                self.best_val_loss = float(previous_best['val_loss'])
                self.best_epoch = int(previous_best['best_epoch'])

        def on_epoch_end(self, epoch, logs=None):
            logs = logs if logs is not None else {}
            kw = {} if hasattr(x_val, 'element_spec') else dict(batch_size=batch_size)
            raw = None
            if ema_optimizer is not None:
                opt = ema_optimizer
                pairs = [(v, opt._model_variables_moving_average[opt._index_dict[opt._var_key(v)]])
                         for v in self.model.trainable_variables]
                raw = [v.numpy() for v, _ in pairs]
                for v, avg in pairs:
                    v.assign(avg)
            logits = self.model.predict(x_val, verbose=0, **kw)
            z = np.asarray(logits, dtype=np.float64).reshape(-1)
            pred = (z > 0.0).astype(np.int8)
            n_err = int((pred != flips_val).sum())
            ler = n_err / n_val
            if ema_optimizer is not None:
                # BCE on logits: softplus(z) - y z
                val_loss = float(np.mean(np.logaddexp(0.0, z) - flips_val * z))
                logs['val_loss_ema'] = val_loss
            else:
                val_loss = float(logs.get('val_loss', float('nan')))
            # in logs so Keras' History records them per epoch
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
                                   selection='min val_errors, tiebreak min val_loss'
                                   + (' (EMA weights)' if ema_optimizer is not None else '')),
                              fh, indent=1)
            if raw is not None:
                for (v, _), w in zip(pairs, raw):
                    v.assign(w)
            print(f"{log_prefix} epoch {epoch + 1}: val_loss={val_loss:.5f} "
                  f"val_ler={ler:.6f} ({n_err:,}/{n_val:,} errors)"
                  f"{'  <- new best' if better else ''}", flush=True)

    return ValLERCheckpoint()
