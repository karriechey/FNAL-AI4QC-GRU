# Created: 2026-09-29 | Last updated: 2026-09-29
"""Stage 1 of the u70 hls4ml baseline: inspect the trained model and evaluate it on the
official evaluation block. Read-only: no file outside this folder is written.

What it does:
  1. Loads gru_u70_reset_after_False.keras with tf_keras (the file was saved with tf_keras 2.15;
     tf.keras 2.15 loads such GRU files without an error but with wrong predictions) and prints
     the architecture.
  2. Rebuilds the same architecture with tf.keras (the object hls4ml parses) and copies the
     weights in, then checks the weights are bit-identical to the 2026-09-23 package
     (model_legacy.h5, run gkd20m_u70_hard_seed0, weights SHA 95d1d3e3...).
  3. Runs the model on the full evaluation block, shots [15,200,000, 17,000,000) of the
     FORMAL d=5 r=5 p=0.004 pool (1,800,000 shots), on CPU, and counts logical errors.
     MWPM errors come from the per-shot MWPM predictions recorded by the training run on the
     same shots.
  4. Checks whether eval_inputs_10000.npy (the inputs used for the hls4ml checks) are the
     first 10,000 shots of that block.

Run from anywhere:  ~/qkeras_preflight/.venv/bin/python stage1_inspect_eval.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import json
import numpy as np
import tensorflow as tf
import tf_keras

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = '/Users/karriechey/Documents/FNAL-AI4QC-GRU/for_giuseppe/gru_models_minimal/gru_u70_reset_after_False.keras'
PKG = '/Users/karriechey/giuseppe_hls4ml/gru_d5_r5_p004_u70_hard_seed0_fp32'
EVAL = '/Users/karriechey/eaf_staging/gru_distill_20m_20260923T043352Z/_eaf_pool/data_d5_p0.004_r5_FORMAL.eval_15200000_17000000.npz'
PER_SHOT = '/Users/karriechey/eaf_staging/gru_distill_20m_20260923T043352Z/runs/gkd20m_u70_hard_seed0/gkd20m_u70_hard_seed0.per_shot_eval.npz'
CPU_REF = '/Users/karriechey/giuseppe_hls4ml/_scratch_step2/cpu_logits_eval_15200000_17000000.npz'
EVAL10K = '/Users/karriechey/Documents/FNAL-AI4QC-GRU/for_giuseppe/gru_models_minimal/eval_inputs_10000.npy'

print('tensorflow', tf.__version__, '| tf_keras', tf_keras.__version__, '| numpy', np.__version__)

# 1. architecture, as stored in the file
src = tf_keras.models.load_model(MODEL, compile=False)
g = [l for l in src.layers if isinstance(l, tf_keras.layers.GRU)]
d = [l for l in src.layers if isinstance(l, tf_keras.layers.Dense)]
assert len(g) == 1 and len(d) == 1, 'expected exactly one GRU and one Dense layer'
g, d = g[0], d[0]
print('\n== architecture (from the .keras file)')
print('layers:', [(l.name, type(l).__name__) for l in src.layers])
print('input shape:', src.input_shape, '| timesteps:', src.input_shape[1], '| features per timestep:', src.input_shape[2])
print(f'GRU: units={g.units} reset_after={g.reset_after} return_sequences={g.return_sequences} '
      f'activation={g.activation.__name__} recurrent_activation={g.recurrent_activation.__name__} use_bias={g.use_bias}')
print('GRU weight shapes:', [w.shape for w in g.get_weights()], '| GRU params:', g.count_params())
print(f'Dense: units={d.units} activation={d.activation.__name__} use_bias={d.use_bias} params={d.count_params()}')
print('output shape:', src.output_shape, '| total params:', src.count_params())
assert g.reset_after is False and g.units == 70 and src.count_params() == 20021

# 2. tf.keras rebuild (what hls4ml parses) + weight identity with the 2026-09-23 package
keras = tf.keras
inp = keras.Input(shape=src.input_shape[1:], name='syndrome_sequence')
h = keras.layers.GRU(g.units, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
                     reset_after=False, return_sequences=False, name='gru')(inp)
out = keras.layers.Dense(1, activation='linear', name='logit')(h)
model = keras.Model(inp, out)
model.get_layer('gru').set_weights(g.get_weights())
model.get_layer('logit').set_weights(d.get_weights())
legacy = keras.models.load_model(os.path.join(PKG, 'model_legacy.h5'), compile=False)
same = all(np.array_equal(a, b) for a, b in zip(model.get_weights(), legacy.get_weights()))
print('\nweights bit-identical to package model_legacy.h5 (run gkd20m_u70_hard_seed0):', same)
assert same

# 3. official evaluation block
sys.path.insert(0, PKG)
from preprocessing import flat_to_sequence
ev = np.load(EVAL)
ps = np.load(PER_SHOT)
ref = np.load(CPU_REF)
x = flat_to_sequence(ev['det_evts'])
n = x.shape[0]
assert int(ev['shot_start']) == 15_200_000 and int(ev['shot_stop']) == 17_000_000 and n == 1_800_000
assert np.array_equal(ps['shot_idx'], np.arange(15_200_000, 17_000_000))
truth = ps['truth'].astype(bool)
assert np.array_equal(truth, ref['truth'].astype(bool))
assert np.array_equal(truth, ev['flips_full'][15_200_000:17_000_000, 0].astype(bool))
logits = model.predict(x, batch_size=20000, verbose=0).reshape(-1)
print('\n== evaluation: FORMAL pool shots [15,200,000, 17,000,000), N =', n)
print('max |logit - 2026-09-23 CPU reference| =', float(np.abs(logits - ref['cpu_logit']).max()))
err_m = int(((logits > 0) != truth).sum())
err_w = int((ps['mwpm_pred'].astype(bool) != truth).sum())
print(f'model errors {err_m}  p_L {err_m / n:.8f}')
print(f'MWPM  errors {err_w}  p_L {err_w / n:.8f}')
print(f'p_L(model)/p_L(MWPM) = {err_m / err_w:.6f}')

# 4. are the 10,000 hls4ml check inputs the first 10,000 shots of this block?
x10k = np.load(EVAL10K)
first = np.array_equal(x10k, x[:10000])
print('\neval_inputs_10000.npy == first 10,000 shots of the block:', first)
json.dump({'n': n, 'model_errors': err_m, 'mwpm_errors': err_w, 'p_L_model': err_m / n, 'p_L_mwpm': err_w / n,
           'ratio': err_m / err_w, 'eval10k_is_first_10000': bool(first)},
          open(os.path.join(HERE, 'stage1_result.json'), 'w'), indent=1)
