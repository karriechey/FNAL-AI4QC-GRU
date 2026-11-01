# Created: 2026-10-28 | Last updated: 2026-10-28
# QAT quantizer specs. Integer bits: weight groups by least squared error of ap_fixed<W, I>
# on the FP32 weights (I from -3 to the profiled value), h and r*h I = 2 and tables I = 1 (the
# smallest that hold [-1, 1] and (-1, 1)), accumulators as profiled (Phase 1) at ACCUM_BITS.
#   bash hpy.sh make_specs.py w8_s16 w6_s16 ...
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcommon as Q
from fq_gru import quantize_np

prof = json.load(open(os.path.join(Q.HERE, 'results', 'p1_profile.json')))['integer_bits']
m = Q.build_fp32()
W, U, b = m.get_layer('gru').get_weights()
wd, bd = m.get_layer('logit').get_weights()
values = {'gru_weight': W, 'gru_recurrent_weight': U, 'gru_bias': b, 'logit_weight': wd, 'logit_bias': bd}


def best_int_bits(v, w, i_max):
    errs = {i: float(((quantize_np(v, w, i) - v) ** 2).sum()) for i in range(-3, i_max + 1)}
    return min(errs, key=errs.get)


def spec_qat(wb, sb):
    spec = {}
    for g in Q.WEIGHT_GROUPS:
        spec[g] = (wb, prof[g] if wb >= 16 else best_int_bits(values[g], wb, prof[g]))
    spec['gru_state'] = (sb, 2)
    spec['gru_table'] = (sb, 1)
    for g in Q.ACCUM_GROUPS:
        spec[g] = (Q.ACCUM_BITS, prof[g])
    return spec


os.makedirs(os.path.join(Q.HERE, 'specs'), exist_ok=True)
for tag in sys.argv[1:]:
    wb, sb = (int(t[1:]) for t in tag.split('_'))
    s = spec_qat(wb, sb)
    Q.dump(s, os.path.join(Q.HERE, 'specs', f'{tag}.json'))
    print(tag, json.dumps(s), flush=True)
