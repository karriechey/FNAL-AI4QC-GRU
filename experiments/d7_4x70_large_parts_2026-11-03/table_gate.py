# Created: 2026-11-03 | Last updated: 2026-11-03
# Paired McNemar between table sizes from the existing full-block hls4ml C-sim logits (Step B of d7_4x70)
#   bash hpy.sh table_gate.py
import json
import os

import numpy as np
from scipy.stats import binomtest

SCRATCH = '/tmp/kchey_qat/d7_4x70'
HERE = os.path.dirname(os.path.abspath(__file__))
truth = np.load(f'{SCRATCH}/eval_d7_15200000_17000000.npz')['truth'].astype(bool)
out = {}
for config in ('P8', 'P16'):
    t4096 = np.load(f'{SCRATCH}/csim_logits_{config}_t4096.npy') > 0
    t1024 = np.load(f'{SCRATCH}/csim_logits_{config}_tdefault.npy') > 0
    wrong_4096, wrong_1024 = t4096 != truth, t1024 != truth
    b, c = int((wrong_1024 & ~wrong_4096).sum()), int((wrong_4096 & ~wrong_1024).sum())
    p = float(binomtest(b, b + c, 0.5).pvalue)
    out[config] = {'n': len(truth), 'errors_t4096': int(wrong_4096.sum()), 'errors_t1024': int(wrong_1024.sum()),
                   'pL_t4096': float(wrong_4096.mean()), 'pL_t1024': float(wrong_1024.mean()),
                   'b_t4096_right_t1024_wrong': b, 'c_t1024_right_t4096_wrong': c, 'p_exact': p, 'pass_p_gt_0.05': p > 0.05}
    print(config, out[config])
os.makedirs(f'{HERE}/results', exist_ok=True)
json.dump(out, open(f'{HERE}/results/table_gate.json', 'w'), indent=1)
