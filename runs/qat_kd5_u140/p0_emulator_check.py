# Created: 2026-10-28 | Last updated: 2026-10-28
# Emulator vs hls4ml C-sim at low-precision points (PTQ, parent weights): generated types and
# decision agreement on the first N eval-block shots.
#   bash hpy.sh p0_emulator_check.py --n 5000 w16_s6 w8_s16
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcommon as Q
import hls_util as H
import numpy as np
import fq_gru as F

ap = argparse.ArgumentParser()
ap.add_argument('tags', nargs='+')
ap.add_argument('--n', type=int, default=5000)
ap.add_argument('--workers', type=int, default=16)
args = ap.parse_args()
d = Q.load_eval(args.n)
x = d['x'].astype(np.float32)
model = Q.build_fp32()
out = {'environment': Q.hls4ml_record()}
for tag in args.tags:
    spec = {k: tuple(v) for k, v in json.load(open(os.path.join(Q.HERE, 'specs', f'{tag}.json'))).items()}
    fq = F.build_fq(spec)
    F.copy_weights(model, fq)
    y_fq = F.predict_logits(fq, x)
    prj = os.path.join(Q.SCRATCH, 'hls', f'p0_emucheck_{tag}')
    hm = Q.convert(model, Q.hls_config(model, spec, reuse_factor=70), prj)
    hm.compile()
    t0 = time.time()
    y = H.csim_predict(hm, x, workers=args.workers)
    out[tag] = {'n': args.n, 'csim_s': round(time.time() - t0, 1), 'types': H.generated_types(prj)['typedefs'],
                'agreement': float(((y > 0) == (y_fq > 0)).mean()), 'max_abs': float(np.abs(y - y_fq).max()),
                'mean_abs': float(np.abs(y - y_fq).mean()),
                'errors_hls': int(((y > 0) != d['truth']).sum()), 'errors_fq': int(((y_fq > 0) != d['truth']).sum())}
    print(tag, json.dumps(out[tag]), flush=True)
Q.dump(out, os.path.join(Q.HERE, 'results', 'p0_emulator_check.json'))
