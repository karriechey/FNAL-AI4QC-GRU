# Created: 2026-10-29 | Last updated: 2026-10-29
# Eval-block inputs for the d7 4 x 70 gate, from the run's own per-shot dump (no pool read, no GPU):
# detection events -> (8, 48) sequences with the training layout, FP32 Keras logits on the CPU, checked
# against the logits the run logged.
#   ~/tfenv/bin/python prep_data.py
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d7_common as C

sys.path.insert(0, '/home/kchey/FNAL-AI4QC-GRU-depth/src')
import numpy as np
import tensorflow as tf
from StudentModels import to_sequence

tf.config.set_visible_devices([], 'GPU')
t0 = time.time()
z = np.load(C.PER_SHOT)
x = to_sequence(z['det_evts'], C.D, C.D, 0.004).astype(np.uint8)
assert x.shape[1:] == (C.N_T, C.N_POS), x.shape
model = C.build_model()
logit = model.predict(x.astype(np.float32), batch_size=20000, verbose=0).reshape(-1)
truth = z['truth'].astype(np.int8)
out = {'n': int(len(truth)), 'tf': tf.__version__, 'device': 'CPU',
       'keras_errors': int(((logit > 0) != truth).sum()), 'logged_errors': int((~z['student_correct']).sum()),
       'mwpm_errors_logged': int((~z['mwpm_correct']).sum()),
       'decision_agreement_with_logged': float(((logit > 0) == (z['student_logit'] > 0)).mean()),
       'max_abs_logit_diff_vs_logged': float(np.abs(logit - z['student_logit']).max()), 's': round(time.time() - t0, 1)}
os.makedirs(C.SCRATCH, exist_ok=True)
np.savez(C.EVAL_NPZ, x=x, truth=truth, keras_logit=logit.astype(np.float32), logged_logit=z['student_logit'],
         mwpm_pred=z['mwpm_pred'], shot_idx=z['shot_idx'])
C.dump(out, os.path.join(C.HERE, 'results', 'prep_data.json'))
print(out, flush=True)
