# Created: 2026-09-29 | Last updated: 2026-09-29
"""Stages 2-4 of the u70 hls4ml baseline: record versions, convert the trained u70 GRU
(reset_after=False) with the hls4ml PR #1568 source tree, and compare the Python C simulation
(hls_model.compile + predict) with Keras on labelled shots of the official evaluation block.

Model and weights are used as they are: no retraining, no quantization-aware changes.
Conversion settings (the baseline):
    backend Vivado, part xcvu13p-flga2577-2-e, clock 5 ns, io_parallel,
    ap_fixed<32,16> default precision (hls4ml defaults for accumulators and activation tables),
    Strategy Latency, ReuseFactor 1, static GRU.

The project also gets a Vivado HLS testbench (tb_data/) of the first N_TB shots, with the Keras
logits as the reference, so a later Vivado HLS run (C sim, synthesis, co-simulation) uses the
same shots.

Usage (Mac; Vivado HLS is not needed for these stages):
    PYTHONPATH=~/Documents/hls4ml-upstream ~/qkeras_preflight/.venv/bin/python \
        stage3_4_convert_csim.py [n_shots]
Writes only inside this folder: prj_u70_pr_fixed32_16_latency_rf1/, stage3_4_result_n<N>.json.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import re
import subprocess
import sys
import time
import numpy as np
import tensorflow as tf
import tf_keras

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
N_TB = 100
PART, CLOCK, PRECISION = 'xcvu13p-flga2577-2-e', 5.0, 'ap_fixed<32,16>'
MODEL = '/Users/karriechey/Documents/FNAL-AI4QC-GRU/for_giuseppe/gru_models_minimal/gru_u70_reset_after_False.keras'
PKG = '/Users/karriechey/giuseppe_hls4ml/gru_d5_r5_p004_u70_hard_seed0_fp32'
EVAL = '/Users/karriechey/eaf_staging/gru_distill_20m_20260923T043352Z/_eaf_pool/data_d5_p0.004_r5_FORMAL.eval_15200000_17000000.npz'
PER_SHOT = '/Users/karriechey/eaf_staging/gru_distill_20m_20260923T043352Z/runs/gkd20m_u70_hard_seed0/gkd20m_u70_hard_seed0.per_shot_eval.npz'
PRJ = os.path.join(HERE, 'prj_u70_pr_fixed32_16_latency_rf1' + ('' if N <= 100000 else f'_n{N}'))

# ---- Stage 2: versions and hls4ml source ------------------------------------------------
import hls4ml
src = os.path.dirname(os.path.dirname(os.path.abspath(hls4ml.__file__)))
git = lambda *a: subprocess.run(['git', '-C', src, *a], capture_output=True, text=True).stdout.strip()
ver = {'hls4ml_import_path': hls4ml.__file__, 'hls4ml_branch': git('branch', '--show-current'),
       'hls4ml_commit': git('rev-parse', 'HEAD'), 'hls4ml_describe': git('describe', '--tags'),
       'hls4ml_worktree_clean': git('status', '--porcelain', '--', 'hls4ml') == '',
       'hls4ml_version_attr': hls4ml.__version__, 'tensorflow': tf.__version__,
       'tf_keras': tf_keras.__version__, 'keras_used_by_hls4ml': tf.keras.__name__, 'numpy': np.__version__,
       'python': sys.version.split()[0], 'vivado_hls': 'not installed on this machine (Mac)'}
print('== versions'); print(json.dumps(ver, indent=1))
assert '/site-packages/' not in hls4ml.__file__, 'hls4ml must come from the PR source tree'
# the fix must be present: the Vivado GRU template reads the reset_after flag
tmpl = open(os.path.join(src, 'hls4ml/templates/vivado/nnet_utils/nnet_recurrent.h')).read()
assert 'reset_after' in tmpl and 'CONFIG_T::reset_after' in tmpl, 'PR #1568 GRU fix not found in template'
print('PR fix present: nnet_recurrent.h reads CONFIG_T::reset_after')

# ---- model (tf_keras file -> tf.keras model with identical weights) ------------------------
srcm = tf_keras.models.load_model(MODEL, compile=False)
g = [l for l in srcm.layers if isinstance(l, tf_keras.layers.GRU)][0]
d = [l for l in srcm.layers if isinstance(l, tf_keras.layers.Dense)][0]
inp = tf.keras.Input(shape=(6, 24), name='syndrome_sequence')
h = tf.keras.layers.GRU(70, activation='tanh', recurrent_activation='sigmoid', reset_after=False,
                        return_sequences=False, name='gru')(inp)
model = tf.keras.Model(inp, tf.keras.layers.Dense(1, name='logit')(h))
model.get_layer('gru').set_weights(g.get_weights())
model.get_layer('logit').set_weights(d.get_weights())
assert model.count_params() == 20021 and model.get_layer('gru').reset_after is False

# ---- labelled shots: the first N of the official evaluation block ---------------------------
sys.path.insert(0, PKG)
from preprocessing import flat_to_sequence
ev = np.load(EVAL)
x = flat_to_sequence(ev['det_evts'][:N]).astype(np.float32)
truth = np.load(PER_SHOT)['truth'][:N].astype(bool)
y_keras = model.predict(x, batch_size=10000, verbose=0).reshape(-1)

# ---- Stage 3: convert ---------------------------------------------------------------------
cfg = hls4ml.utils.config_from_keras_model(model, granularity='name', default_precision=PRECISION, backend='Vivado')
cfg['Model']['Strategy'] = 'Latency'
cfg['Model']['ReuseFactor'] = 1
for name in ('gru', 'logit'):
    cfg['LayerName'][name]['Strategy'] = 'Latency'
    cfg['LayerName'][name]['ReuseFactor'] = 1
cfg['LayerName']['gru']['static'] = True
print('\n== hls4ml config'); print(json.dumps(cfg, indent=1, default=str))
os.makedirs(PRJ, exist_ok=True)
tbx, tby = os.path.join(PRJ, 'tb_x.npy'), os.path.join(PRJ, 'tb_y.npy')
np.save(tbx, x[:N_TB].reshape(N_TB, -1)); np.save(tby, y_keras[:N_TB].reshape(N_TB, 1))
hm = hls4ml.converters.convert_from_keras_model(
    model, hls_config=cfg, output_dir=PRJ, project_name='myproject', backend='Vivado',
    io_type='io_parallel', part=PART, clock_period=CLOCK, input_data_tb=tbx, output_data_tb=tby)
hm.write()
gl = [l for l in hm.get_layers() if l.class_name == 'GRU'][0]
layer_info = {k: str(gl.get_attr(k)) for k in ('apply_reset_gate', 'strategy', 'reuse_factor', 'recurrent_reuse_factor', 'static')}
layer_info.update({f'weight_{k}_shape': list(np.shape(w.data)) for k, w in gl.weights.items()})
params_h = open(os.path.join(PRJ, 'firmware', 'parameters.h')).read()
layer_info['generated_reset_after'] = re.findall(r'static const bool reset_after = (\w+);', params_h)
print('\n== hls4ml GRU layer'); print(json.dumps(layer_info, indent=1))
for l in hm.get_layers():
    print(f'  {l.name:18s} {l.class_name:10s}', {k: str(v.precision) for k, v in l.get_layer_precision().items()})

# ---- Stage 4: Python C simulation vs Keras -----------------------------------------------------
t0 = time.time(); hm.compile(); t_compile = time.time() - t0
t0 = time.time(); y_hls = np.asarray(hm.predict(np.ascontiguousarray(x)), float).reshape(-1); t_pred = time.time() - t0
diff = np.abs(y_hls - y_keras)
pk, ph = y_keras > 0, y_hls > 0
diff_idx = np.flatnonzero(pk != ph)
err_k, err_h = int((pk != truth).sum()), int((ph != truth).sum())
res = {'versions': ver, 'config': cfg, 'gru_layer': layer_info, 'n': N,
       'shots': [15_200_000, 15_200_000 + N], 'max_abs_diff': float(diff.max()), 'mean_abs_diff': float(diff.mean()),
       'decision_agreement': float((pk == ph).mean()), 'decisions_differ': int(diff_idx.size),
       'errors_keras': err_k, 'errors_hls_csim': err_h, 'p_L_keras': err_k / N, 'p_L_hls_csim': err_h / N,
       'differing_shots': [{'shot': int(15_200_000 + i), 'keras_logit': float(y_keras[i]), 'hls_logit': float(y_hls[i]),
                            'truth': bool(truth[i]), 'keras_correct': bool(pk[i] == truth[i]),
                            'hls_correct': bool(ph[i] == truth[i])} for i in diff_idx],
       'compile_s': round(t_compile, 1), 'predict_s': round(t_pred, 1)}
print('\n== C simulation (Python, hls4ml compile + predict) vs Keras')
for k in ('n', 'shots', 'max_abs_diff', 'mean_abs_diff', 'decision_agreement', 'decisions_differ',
          'errors_keras', 'errors_hls_csim', 'p_L_keras', 'p_L_hls_csim', 'compile_s', 'predict_s'):
    print(f'  {k}: {res[k]}')
for s in res['differing_shots']:
    print('  differs:', s)
np.save(os.path.join(HERE, f'hls_csim_logits_first{N}.npy'), y_hls)
json.dump(res, open(os.path.join(HERE, f'stage3_4_result_n{N}.json'), 'w'), indent=1, default=str)
