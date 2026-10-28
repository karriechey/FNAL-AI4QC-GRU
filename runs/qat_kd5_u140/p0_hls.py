# Created: 2026-10-28 | Last updated: 2026-10-28
# Phase 0.3-0.4 on the host CPU:
#   wiring  ap_fixed<32,10,RND,SAT> everywhere: FQ emulator vs FP32 on the whole eval block,
#           hls4ml C-sim vs FQ emulator and FP32 on the first N shots
#   QGRU    QKeras QGRU of the same weights: does hls4ml give the same layer config and weights as
#           the Keras GRU with the same per-layer precisions?
#   ~/hls4ml_main_env/bin/python p0_hls.py --n-csim 20000
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcommon as Q
import hls_util as H

import numpy as np
import tensorflow as tf
import fq_gru as F

ap = argparse.ArgumentParser()
ap.add_argument('--n-csim', type=int, default=20000)
ap.add_argument('--workers', type=int, default=32)
ap.add_argument('--rf', type=int, default=70)
args = ap.parse_args()

out = {'environment': Q.hls4ml_record(), 'tf': tf.__version__}
d = Q.load_eval()
x, truth = d['x'].astype(np.float32), d['truth'].astype(bool)
fp = d['fp32_logit']
model = Q.build_fp32()

wide = {g: (32, 10) for g in Q.HLS_KEYS}
fq = F.build_fq(wide)
F.copy_weights(model, fq)
t0 = time.time()
y_fq = F.predict_logits(fq, x)
out['wiring_fq_vs_fp32'] = {**Q.score(y_fq > 0, d), 'max_abs_logit_diff': float(np.abs(y_fq - fp).max()),
                            'decision_agreement': float(((y_fq > 0) == (fp > 0)).mean()), 's': round(time.time() - t0, 1)}
print('FQ<32,10> vs FP32', out['wiring_fq_vs_fp32'], flush=True)

n = args.n_csim
prj = os.path.join(Q.SCRATCH, 'hls', 'p0_wide_32_10')
cfg = Q.hls_config(model, wide, reuse_factor=args.rf)
hm = Q.convert(model, cfg, prj)
hm.compile()
t0 = time.time()
y_hls = H.csim_predict(hm, x[:n], workers=args.workers)
br = H.weights_txt(prj, 'br2')
out['wiring_hls'] = {'n': n, 'csim_s': round(time.time() - t0, 1), 'types': H.generated_types(prj),
                     'recurrent_bias_all_zero': bool((br == 0).all()), 'recurrent_bias_len': int(len(br)),
                     'hls_vs_fq_agreement': float(((y_hls > 0) == (y_fq[:n] > 0)).mean()),
                     'hls_vs_fq_max_abs': float(np.abs(y_hls - y_fq[:n]).max()),
                     'hls_vs_fp32_agreement': float(((y_hls > 0) == (fp[:n] > 0)).mean()),
                     'hls_vs_fp32_max_abs': float(np.abs(y_hls - fp[:n]).max()),
                     'errors_hls': int(((y_hls > 0) != truth[:n]).sum()),
                     'errors_fq': int(((y_fq[:n] > 0) != truth[:n]).sum()),
                     'errors_fp32': int(((fp[:n] > 0) != truth[:n]).sum())}
print('hls wiring', out['wiring_hls'], flush=True)
np.save(os.path.join(Q.SCRATCH, 'hls', 'p0_wide_hls_logits.npy'), y_hls.astype(np.float32))

# QGRU path: 8-bit weights, everything else as the Keras-GRU conversion
try:
    from StudentModels import build_gru_student
    from qkeras import quantized_bits
    from qkeras.qrecurrent import QGRU
    from qkeras import QDense
    qw = quantized_bits(8, 6, alpha=1)   # ap_fixed<8,7>: QKeras integer bits exclude the sign
    inp = tf.keras.Input(shape=(Q.N_T, Q.N_POS), name='syndrome_sequence')
    h = QGRU(Q.UNITS, reset_after=False, kernel_quantizer=qw, recurrent_quantizer=qw, bias_quantizer=qw,
             activation='tanh', recurrent_activation='sigmoid', name='gru')(inp)
    qm = tf.keras.Model(inp, QDense(1, kernel_quantizer=qw, bias_quantizer=qw, name='logit')(h))
    qm.get_layer('gru').set_weights(model.get_layer('gru').get_weights())
    qm.get_layer('logit').set_weights(model.get_layer('logit').get_weights())
    spec8 = {**wide, **{g: (8, 7) for g in Q.WEIGHT_GROUPS}}
    prj_q = os.path.join(Q.SCRATCH, 'hls', 'p0_qgru_w8')
    prj_k = os.path.join(Q.SCRATCH, 'hls', 'p0_kgru_w8')
    hq = Q.convert(qm, Q.hls_config(qm, spec8, reuse_factor=args.rf), prj_q)
    hk = Q.convert(model, Q.hls_config(model, spec8, reuse_factor=args.rf), prj_k)
    tq, tk = H.generated_types(prj_q), H.generated_types(prj_k)
    w_keras = model.get_layer('gru').get_weights()[0].reshape(-1)
    wq_q, wq_k = H.weights_txt(prj_q, 'w2'), H.weights_txt(prj_k, 'w2')
    w_ref = F.quantize_np(np.sort(w_keras), 8, 7)
    out['qgru_check'] = {
        'same_typedefs': tq['typedefs'] == tk['typedefs'], 'same_reset_after': tq['reset_after'] == tk['reset_after'],
        'qgru_types': tq, 'kgru_types': tk,
        'qgru_weights_equal_keras_gru_conversion': bool(np.array_equal(np.sort(wq_q), np.sort(wq_k))),
        'qgru_weight_file_is_raw_float': bool(np.allclose(np.sort(wq_q), np.sort(w_keras), atol=1e-6)),
        'qgru_weight_file_on_8bit_grid': bool(np.allclose(np.sort(wq_q), w_ref, atol=1e-9)),
        'qgru_layer_attrs': {k: str(v) for k, v in hq.graph['gru'].attributes.items()
                             if 'quantizer' in k or k in ('apply_reset_gate', 'activation', 'recurrent_activation')},
    }
except Exception as e:
    out['qgru_check'] = f'failed: {type(e).__name__}: {e}'
print('qgru', out['qgru_check'], flush=True)
Q.dump(out, os.path.join(Q.HERE, 'results', 'p0_hls.json'))
