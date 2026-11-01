# Created: 2026-10-29 | Last updated: 2026-10-29
# Activation-table ablation for the d7 4 x 70 stacked GRU, everything else as Step A (ap_fixed<32,16>):
# table_size {1024, 4096} x table_t {ap_fixed<18,8> (hls4ml default), ap_fixed<32,16>}, plus ap_fixed<18,1>
# (same width as the default, the 7 unused integer bits moved to the fraction; values are in (-1, 1)).
# Per-layer trace error vs Keras sub-models on the first 2,000 eval-block shots; table storage per design.
#   bash hpy.sh table_ablation.py
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d7_common as C
import numpy as np
import tensorflow as tf

N = 2000
VARIANTS = [('t1024_18_8_default', 1024, None), ('t4096_18_8', 4096, None),
            ('t1024_32_16', 1024, 'ap_fixed<32,16>'), ('t4096_32_16', 4096, 'ap_fixed<32,16>'),
            ('t1024_18_1_extra', 1024, 'ap_fixed<18,1>'), ('t4096_18_1_extra', 4096, 'ap_fixed<18,1>')]
d = C.load_eval(N)
x = np.ascontiguousarray(d['x'].astype(np.float32))
kl = d['keras_logit'].astype(float)
model = C.build_model()
names = C.GRU_NAMES + ['logit']
ref = {n: tf.keras.Model(model.input, model.get_layer(n).output).predict(x, batch_size=2000, verbose=0) for n in names}
out = {'environment': C.hls4ml_record(), 'n': N, 'variants': {}}
for label, table_size, table_t in VARIANTS:
    cfg = C.base_config(model)
    for n in C.GRU_NAMES:
        cfg['LayerName'][n]['Trace'] = True
        cfg['LayerName'][n]['table_size'] = table_size
        if table_t:
            cfg['LayerName'][n]['table_t'] = table_t
    cfg['LayerName']['logit']['Trace'] = True
    prj = os.path.join(C.SCRATCH, f'hls_ablation_{label}')
    hm = C.convert(model, cfg, prj)
    hm.compile()
    y, tr = hm.trace(x)
    params = open(os.path.join(prj, 'firmware', 'parameters.h')).read()
    defines = open(os.path.join(prj, 'firmware', 'defines.h')).read()
    tsz = sorted(set(re.findall(r'table_size = (\d+);', params)))
    ttype = sorted(set(re.findall(r'typedef (ap_fixed<[^>]+>) gru\w*_table_t;', defines)))
    width = int(re.match(r'ap_fixed<(\d+)', ttype[0]).group(1))
    # Resource GRU cell, one unit in parallel at RF 70: 2 sigmoid + 1 tanh table per layer, 4 layers
    n_tables = 3 * len(C.GRU_NAMES)
    rec = {'table_size': tsz, 'table_t': ttype, 'n_tables': n_tables,
           'table_bits_total': n_tables * int(tsz[0]) * width,
           'table_bram18k_if_in_bram': n_tables * -(-int(tsz[0]) * width // 18432)}
    for n in names:
        e = np.abs(np.asarray(tr[n], float).reshape(N, -1) - np.asarray(ref[n], float).reshape(N, -1))
        rec[n] = {'mean_abs_err': float(e.mean()), 'p99_abs_err': float(np.quantile(e, 0.99)), 'max_abs_err': float(e.max())}
    y = np.asarray(y, float).reshape(-1)
    flips = (y > 0) != (kl > 0)
    rec['decision_flips'] = int(flips.sum())
    rec['flips_abs_keras_logit_ge_0p2'] = int((flips & (np.abs(kl) >= 0.2)).sum())
    out['variants'][label] = rec
    print(label, {k: rec[k] for k in ('table_size', 'table_t', 'table_bits_total', 'decision_flips')},
          'logit mean err', round(rec['logit']['mean_abs_err'], 4), 'gru_3 mean err', round(rec['gru_3']['mean_abs_err'], 5), flush=True)
C.dump(out, os.path.join(C.HERE, 'results', 'table_ablation.json'))
