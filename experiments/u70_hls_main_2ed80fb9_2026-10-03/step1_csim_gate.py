# Created: 2026-10-03 | Last updated: 2026-10-03
"""Step 1: C-simulation gate for the u70 GRU on hls4ml main 2ed80fb9, ap_fixed<32,16>.

Converts the trained model (default Resource strategy, ReuseFactor 70 (--rf), static GRU, io_parallel,
xcvu13p, 5 ns), compiles the C++ with g++ (hls_model.compile) and compares
hls_model.predict with Keras on identical shots of the evaluation block.

Gate (from the plan): decision match must be 100% on the first 10,000 shots and the
logical-error-rate shift must be within noise (exact McNemar p >= 0.05). The script exits
with status 1 if either fails.

With --n larger than 10,000 the extra shots are run in parallel worker processes as a
higher-statistics check; the gate itself is always decided on the first 10,000.

Usage (grace1):
    ~/hls4ml_main_env/bin/python step1_csim_gate.py [--n 10000] [--workers 48]
Writes: results/step1_csim_gate_<strategy>_rf<RF>_n<N>.json, results/step1_hls_logits_<...>.npy,
        hls_step1_fixed32_16_<strategy>_rf<RF>/ (generated project, gitignored)

Note (2026-10-03): with Resource and ReuseFactor 70, hls4ml main 2ed80fb9 generates the GRU
recurrent multiply (70 x 210) with reuse_factor 70 but the kernel for RF > n_in
(DenseResource_rf_gt_nin_rem0), because recurrent_templates.py chooses the kernel from the
layer's input-multiply reuse factor (snapped to 72). The C simulation stops on that kernel's
assert. See patch_recurrent_kernel_choice.py for the one-line local fix used for comparison.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

import u70_common as U

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=10000)
ap.add_argument('--workers', type=int, default=48)
ap.add_argument('--rf', type=int, default=70)
ap.add_argument('--strategy', default='Resource')
args = ap.parse_args()
GATE_N = 10000

env = U.environment_record()
print(json.dumps(env, indent=1))
model = U.build_model()
d = U.load_eval(args.n)
x, truth = d['x'], d['truth']
y_keras = model.predict(x, batch_size=10000, verbose=0).reshape(-1)
# The rebuilt model must reproduce the logits the EAF run stored for the same shots.
keras_vs_eaf = float(np.abs(y_keras - d['keras_logit_eaf']).max())
assert ((y_keras > 0) == (d['keras_logit_eaf'] > 0)).all(), 'rebuilt Keras model disagrees with the stored EAF logits'

cfg = U.base_config(model, 'ap_fixed<32,16>', args.strategy, args.rf)
tag = f'{args.strategy.lower()}_rf{args.rf}' + U.patch_tag()
prj = os.path.join(U.HERE, f'hls_step1_fixed32_16_{tag}')
hm = U.convert(model, cfg, prj)
t0 = time.time(); hm.compile(); t_compile = time.time() - t0


def work(bounds):
    a, b = bounds
    return a, U.csim_predict(hm, x[a:b])


t0 = time.time()
chunks = [(a, min(a + 2500, len(x))) for a in range(0, len(x), 2500)]
y_hls = np.empty(len(x))
if len(chunks) == 1:
    y_hls[:] = U.csim_predict(hm, x)
else:
    with mp.get_context('fork').Pool(min(args.workers, len(chunks))) as pool:
        for a, y in pool.imap_unordered(work, chunks):
            y_hls[a:a + len(y)] = y
t_pred = time.time() - t0

pk, ph = y_keras > 0, y_hls > 0
res = {'label': U.RESULT_LABEL, 'step': 1, 'environment': env, 'config': cfg,
       'shots': [U.EVAL_START, U.EVAL_START + len(x)], 'partition': 'FORMAL eval block, EAF x86 pool (non-sealed)',
       'keras_rebuild_vs_eaf_max_abs_logit_diff': keras_vs_eaf,
       'compile_s': round(t_compile, 1), 'predict_s': round(t_pred, 1)}
for n in sorted({GATE_N, len(x)}):
    s = U.paired_stats(pk[:n], ph[:n], truth[:n])
    s['logit_exact_match_rate'] = float((y_hls[:n] == y_keras[:n]).mean())
    s['max_abs_logit_diff'] = float(np.abs(y_hls[:n] - y_keras[:n]).max())
    s['mean_abs_logit_diff'] = float(np.abs(y_hls[:n] - y_keras[:n]).mean())
    flip = np.flatnonzero(pk[:n] != ph[:n])
    s['flipped_shots_max_abs_keras_logit'] = float(np.abs(y_keras[flip]).max()) if flip.size else None
    res[f'n{n}'] = s
g = res[f'n{GATE_N}']
res['gate_pass'] = bool(g['decision_match_rate'] == 1.0 and g['p_exact'] >= 0.05)
os.makedirs(os.path.join(U.HERE, 'results'), exist_ok=True)
res['reuse_factors'] = U.generated_reuse_factors(prj)
np.save(os.path.join(U.HERE, 'results', f'step1_hls_logits_{tag}_n{len(x)}.npy'), y_hls)
json.dump(res, open(os.path.join(U.HERE, 'results', f'step1_csim_gate_{tag}_n{len(x)}.json'), 'w'), indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k not in ('config', 'environment')}, indent=1, default=str))
print('STEP1 GATE', 'PASS' if res['gate_pass'] else 'FAIL')
sys.exit(0 if res['gate_pass'] else 1)
