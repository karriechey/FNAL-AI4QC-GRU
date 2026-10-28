# Created: 2026-10-28 | Last updated: 2026-10-28
# Phase 1: value ranges of every precision group on the whole eval block (float model),
# integer bits = minimum signed width for max|v| + 1 headroom, then the two 1-D PTQ sweeps
# on the hls4ml emulator (fq_gru). GPU container: bash pod.sh qat_p1 python3 p1_ptq.py
import csv
import json
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
import fq_gru as F

RES = os.path.join(Q.HERE, 'results')
LOGITS = os.path.join(Q.SCRATCH, 'logits')
os.makedirs(LOGITS, exist_ok=True)
WEIGHT_AXIS = [16, 12, 10, 8, 6, 5, 4]
STATE_AXIS = [16, 12, 10, 8, 6]

d = Q.load_eval()
x = d['x'].astype(np.float32)
model = Q.build_fp32()

t0 = time.time()
fl = F.build_fq({}, tables=False)
F.copy_weights(model, fl)
taps_fn = tf.function(lambda xb: fl.get_layer('gru')(xb, return_internals=True))
maxabs, pct = {}, {}
samples = {}
for a in range(0, len(x), 50000):
    h, taps = taps_fn(tf.convert_to_tensor(x[a:a + 50000]))
    for t, tp in enumerate(taps):
        for k, v in tp.items():
            v = np.abs(v.numpy())
            maxabs[k] = max(maxabs.get(k, 0.0), float(v.max()))
            if a == 0:
                samples.setdefault(k, []).append(v.reshape(-1)[::7])
lg = F.predict_logits(fl, x)
assert np.array_equal(lg, d['fp32_logit'])
for k, v in samples.items():
    v = np.concatenate(v)
    pct[k] = {'p99.9': float(np.quantile(v, 0.999)), 'p99.99': float(np.quantile(v, 0.9999))}

W, U, b = model.get_layer('gru').get_weights()
wd, bd = model.get_layer('logit').get_weights()
ranges = {
    'gru_weight': float(np.abs(W).max()),
    'gru_recurrent_weight': float(np.abs(U).max()),
    'gru_bias': float(np.abs(b).max()),
    'logit_weight': float(np.abs(wd).max()),
    'logit_bias': float(np.abs(bd).max()),
    'gru_accum': max(maxabs[k] for k in ('xin', 'rec_zr', 'pre_zr', 'zr', 'rec_h', 'pre_h', 'hh')),
    'gru_state': max(maxabs['h'], maxabs['h_reset']),
    'gru_table': 1.0 - 1e-6,   # sigmoid and tanh table entries are inside (-1, 1)
    'logit_accum': float(np.abs(lg).max()),
}
int_bits = {k: Q.int_bits_from_max(v) for k, v in ranges.items()}
weight_pct = {k: {'p99': float(np.quantile(np.abs(a), 0.99)), 'p99.9': float(np.quantile(np.abs(a), 0.999))}
              for k, a in (('gru_weight', W), ('gru_recurrent_weight', U), ('gru_bias', b), ('logit_weight', wd))}
prof = {'shots': [Q.EVAL_START, Q.EVAL_STOP], 'rule': 'I = floor(log2 max|v|) + 2 (sign included) + 1 headroom',
        'max_abs': ranges, 'integer_bits': int_bits, 'internal_max_abs': maxabs, 'internal_percentiles_50k': pct,
        'weight_percentiles': weight_pct, 'profile_s': round(time.time() - t0, 1)}
Q.dump(prof, os.path.join(RES, 'p1_profile.json'))
print(json.dumps(prof['integer_bits']), json.dumps(ranges), flush=True)

rows = []
points = [('weights', w, 16) for w in WEIGHT_AXIS] + [('state', 16, s) for s in STATE_AXIS]
for axis, wb, sb in points:
    spec = Q.spec_for(int_bits, wb, sb)
    m = F.build_fq(spec)
    F.copy_weights(model, m)
    t1 = time.time()
    y = F.predict_logits(m, x)
    sc = Q.score(y > 0, d)
    mc = sc['mcnemar_vs_fp32']
    row = {'method': 'PTQ', 'axis': axis, 'weight_bits': wb, 'state_bits': sb, 'accum_bits': Q.ACCUM_BITS,
           'total_bits': wb + sb, 'ler': sc['ler'], 'ler_fp32': sc['ler_fp32'], 'ler_mwpm': sc['ler_mwpm'],
           'ler_over_fp32': sc['ler_over_fp32'], 'ler_over_mwpm': sc['ler_over_mwpm'],
           'b': mc['b'], 'c': mc['c'], 'n_disc': mc['n_disc'], 'p_exact': mc['p_exact'],
           'agreement_fp32': mc['agreement'], 'acceptable': Q.acceptable(sc),
           'spec': json.dumps(spec), 's': round(time.time() - t1, 1)}
    rows.append(row)
    np.save(os.path.join(LOGITS, f'ptq_w{wb}_s{sb}.npy'), y.astype(np.float32))
    print(json.dumps({k: v for k, v in row.items() if k != 'spec'}), flush=True)

with open(os.path.join(RES, 'p1_ptq.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def knee(axis_rows, key):
    # smallest width that is acceptable and every wider swept width on this axis is too
    good = None
    for r in sorted(axis_rows, key=lambda r: -r[key]):
        if not r['acceptable']:
            break
        good = r[key]
    return good


knees = {'weights': knee([r for r in rows if r['axis'] == 'weights'], 'weight_bits'),
         'state': knee([r for r in rows if r['axis'] == 'state'], 'state_bits')}
Q.dump(knees, os.path.join(RES, 'p1_knees.json'))
print('knees', knees, flush=True)
