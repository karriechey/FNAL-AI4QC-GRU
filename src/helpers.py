# Created: 2026-09-25
# Last modified: 2026-09-25
"""Small shared helpers for the GRU pipeline.

The first four functions are moved verbatim from the old repo's RCNN files so that no RCNN
code has to come across with them:

  set_seeds, learning_rate_scheduler (+ LR_FLOOR), lookup_mwpm
      from FNAL-QCDecoding-FPGA/train_one.py @ d5ee86f
  mcnemar_from_correct
      from FNAL-QCDecoding-FPGA/eval_on_tail.py @ d5ee86f

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
            logits = self.model.predict(x_val, batch_size=batch_size, verbose=0)
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
