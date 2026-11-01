# Created: 2026-10-29 | Last updated: 2026-10-29
# Step A diagnosis: where does the C-sim leave Keras? Per-layer trace vs Keras sub-models on the first 2,000
# shots, for the Step A configuration and with finer activation tables (table_size 4096, table_t <32,16>).
# Table error that grows layer by layer points at the lookup tables; a jump at one layer points at a bug.
#   bash hpy.sh step_a_diag.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d7_common as C
import numpy as np
import tensorflow as tf

N = 2000
d = C.load_eval(N)
x = np.ascontiguousarray(d['x'].astype(np.float32))
model = C.build_model()
names = C.GRU_NAMES + ['logit']
ref = {n: tf.keras.Model(model.input, model.get_layer(n).output).predict(x, batch_size=2000, verbose=0) for n in names}
out = {}
for label, table_size, table_t in (('step_a', None, None), ('tables_4096_32_16', 4096, 'ap_fixed<32,16>')):
    cfg = C.base_config(model)
    for n in C.GRU_NAMES:
        cfg['LayerName'][n]['Trace'] = True
        if table_size:
            cfg['LayerName'][n]['table_size'] = table_size
            cfg['LayerName'][n]['table_t'] = table_t
    cfg['LayerName']['logit']['Trace'] = True
    hm = C.convert(model, cfg, os.path.join(C.SCRATCH, f'hls_diag_{label}'))
    hm.compile()
    y, tr = hm.trace(x)
    rec = {}
    for n in names:
        e = np.abs(np.asarray(tr[n], float).reshape(N, -1) - np.asarray(ref[n], float).reshape(N, -1))
        rec[n] = {'max_abs_err': float(e.max()), 'mean_abs_err': float(e.mean()),
                  'p99_abs_err': float(np.quantile(e, 0.99))}
    y = np.asarray(y, float).reshape(-1)
    kl = d['keras_logit'].astype(float)
    rec['decision_flips'] = int(((y > 0) != (kl > 0)).sum())
    rec['flips_with_abs_keras_logit_ge_0p2'] = int((((y > 0) != (kl > 0)) & (np.abs(kl) >= 0.2)).sum())
    out[label] = rec
    print(label, rec, flush=True)
C.dump(out, os.path.join(C.HERE, 'results', 'step_a_diag.json'))
