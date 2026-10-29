# Created: 2026-10-29 | Last updated: 2026-10-29
# Step A: does hls4ml convert and C-simulate the stacked GRU? ap_fixed<32,16> default precision (tables at
# the hls4ml default), Resource, io_parallel, static GRU, RF 70, xczu49dr, shots [15.2M, 15.21M).
# Pass: decisions equal to Keras except shots with |Keras logit| < 0.2.
#   bash hpy.sh step_a.py
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d7_common as C
import numpy as np

out = {'environment': C.hls4ml_record(), 'shots': [C.EVAL_START, C.EVAL_START + 10_000]}
d = C.load_eval(10_000)
x, truth, kl = d['x'].astype(np.float32), d['truth'].astype(bool), d['keras_logit'].astype(np.float64)
model = C.build_model()
prj = os.path.join(C.SCRATCH, 'hls_step_a_fixed32_16')
try:
    cfg = C.base_config(model)
    hm = C.convert(model, cfg, prj)
    out['layer_types'] = {n: hm.graph[n].class_name for n in hm.graph}
    out['gru_attrs'] = {n: {k: str(hm.graph[n].get_attr(k)) for k in ('return_sequences', 'apply_reset_gate', 'n_unit_par',
                                                                        'n_cycles', 'reuse_factor', 'static')}
                        for n in C.GRU_NAMES}
    hm.compile()
except Exception as e:
    out['status'] = 'FAILED in conversion or compile'
    out['error'] = f'{type(e).__name__}: {e}'
    out['traceback'] = traceback.format_exc()
    C.dump(out, os.path.join(C.HERE, 'results', 'step_a.json'))
    print(out['traceback'], flush=True)
    sys.exit(1)

t0 = time.time()
y = C.csim_predict(hm, x)
out['csim_s'] = round(time.time() - t0, 1)
flip = np.flatnonzero((y > 0) != (kl > 0))
out['n_flipped'] = int(len(flip))
out['flipped'] = [{'shot': int(d['shot_idx'][i]), 'keras_logit': float(kl[i]), 'hls_logit': float(y[i]),
                   'truth': int(truth[i])} for i in flip]
out['flipped_outside_0p2'] = int((np.abs(kl[flip]) >= 0.2).sum())
out['max_abs_logit_diff'] = float(np.abs(y - kl).max())
out['mean_abs_logit_diff'] = float(np.abs(y - kl).mean())
out['stats_vs_keras'] = C.paired(kl > 0, y > 0, truth)
out['status'] = 'PASS' if out['flipped_outside_0p2'] == 0 else 'FAIL: flips with |Keras logit| >= 0.2'
np.save(os.path.join(C.SCRATCH, 'step_a_hls_logits.npy'), y.astype(np.float32))
C.dump(out, os.path.join(C.HERE, 'results', 'step_a.json'))
print({k: v for k, v in out.items() if k not in ('flipped', 'traceback')}, flush=True)
for f in out['flipped']:
    print('flipped', f, flush=True)
