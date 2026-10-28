# Created: 2026-10-28 | Last updated: 2026-10-28
# Phase 0: eval-block arrays, MWPM decoded on the same shots, FP32 parent rescored (TF32 off),
# compared with the recorded numbers and per-shot dump; the float FQ model must equal Keras.
# Runs in the qdec:tf215 container (GPU, stim, pymatching).
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcommon as Q

import tensorflow as tf
tf.config.experimental.enable_tensor_float_32_execution(False)
for g in tf.config.list_physical_devices('GPU'):
    tf.config.set_logical_device_configuration(g, [tf.config.LogicalDeviceConfiguration(memory_limit=6000)])

import pool_rows as PR
import fq_gru as F
from StudentModels import to_sequence

out = {'tf': tf.__version__, 'gpu': [d.name for d in tf.config.list_logical_devices('GPU')]}
t0 = time.time()
lo, hi = Q.EVAL_START, Q.EVAL_STOP
det = PR.rows(Q.POOL, 'det_evts', lo, hi).astype(np.uint8)
truth = PR.rows(Q.POOL, 'flips', lo, hi).astype(np.int8).reshape(-1)
x = to_sequence(det, Q.D, Q.ROUNDS, Q.P).astype(np.uint8)

import pymatching
from circuit_generators import build_circuit
circ = build_circuit(Q.D, Q.P, Q.ROUNDS)
pym = pymatching.Matching.from_detector_error_model(circ.detector_error_model(decompose_errors=True))
mwpm = pym.decode_batch(det, bit_packed_predictions=False, bit_packed_shots=False).astype(np.int8).reshape(-1)

model = Q.build_fp32()
logit = F.predict_logits(model, x.astype(np.float32))
out['data_s'] = round(time.time() - t0, 1)

rec = np.load(Q.PARENT_PER_SHOT)
assert (rec['shot_idx'] == np.arange(lo, hi)).all()
out['truth_equal_recorded'] = bool((rec['truth'] == truth).all())
out['mwpm_equal_recorded'] = bool((rec['mwpm_pred'] == mwpm).all())
out['fp32_decisions_equal_recorded'] = float(((rec['student_logit'] > 0) == (logit > 0)).mean())
out['fp32_logit_max_abs_diff_recorded'] = float(np.abs(rec['student_logit'] - logit).max())
errs, mw_errs = int(((logit > 0) != truth).sum()), int((mwpm != truth).sum())
out.update({'n': int(hi - lo), 'fp32_errors': errs, 'mwpm_errors': mw_errs, 'fp32_p_L': errs / (hi - lo),
            'mwpm_p_L': mw_errs / (hi - lo), 'ratio': errs / mw_errs, 'recorded': Q.RECORDED,
            'matches_recorded': round(errs / (hi - lo), 6) == Q.RECORDED['p_L']
                                and round(mw_errs / (hi - lo), 6) == Q.RECORDED['mwpm_p_L']})

# float FQ model (no quantizer, exact activations) against Keras
fq = F.build_fq({}, tables=False)
F.copy_weights(model, fq)
y = F.predict_logits(fq, x[:200000].astype(np.float32))
out['fq_float_vs_keras_200k'] = {'max_abs_diff': float(np.abs(y - logit[:200000]).max()),
                                 'decision_agreement': float(((y > 0) == (logit[:200000] > 0)).mean())}

os.makedirs(Q.DATA, exist_ok=True)
np.savez(Q.EVAL_NPZ, x=x, truth=truth, mwpm_pred=mwpm, fp32_logit=logit.astype(np.float32),
         shot_idx=np.arange(lo, hi, dtype=np.int64))
Q.dump(out, os.path.join(Q.HERE, 'results', 'p0_reproduce.json'))
print(out, flush=True)
