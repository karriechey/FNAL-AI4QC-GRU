# Created: 2026-10-29 | Last updated: 2026-10-29
# Step B for the d7 4 x 70 stacked GRU:
#   1 integer bits per layer and type from a float64 numpy replay over all 1.8M eval-block shots,
#     I = floor(log2 max|v|) + 2
#   2 full-block C-sim of P16 (every type 16 bits) and P8 (weights and biases 8 bits; accumulators, state,
#     tables and input 16 bits), all AP_RND / AP_SAT, each with the chosen tables (--table-size, table_t
#     16 bits) and with the hls4ml default tables (1024 x ap_fixed<18,8>) as baseline; errors vs FP32 Keras
#   3 the four Vivado-backend projects with a 100-shot testbench (inputs, expected = this C-sim) for EAF
#   bash hpy.sh step_b.py --table-size 4096
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d7_common as C
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--table-size', type=int, required=True)
ap.add_argument('--workers', type=int, default=64)
ap.add_argument('--n', type=int, default=None)
args = ap.parse_args()

d = C.load_eval(args.n)
x, truth, kl = d['x'].astype(np.float32), d['truth'].astype(bool), d['keras_logit'].astype(np.float64)
model = C.build_model()
out = {'environment': C.hls4ml_record(), 'n': int(len(truth)), 'table_size_chosen': args.table_size}


def replay_ranges(model, x, chunk=100_000):
    # max |value| of every quantity a type holds, per GRU layer (reset_after=False, gates z, r, h)
    sig = lambda v: 1.0 / (1.0 + np.exp(-v))
    m = {}
    upd = lambda k, v: m.__setitem__(k, max(m.get(k, 0.0), float(np.abs(v).max())))
    wts = {n: [w.astype(np.float64) for w in model.get_layer(n).get_weights()] for n in C.GRU_NAMES}
    wd, bd = [w.astype(np.float64) for w in model.get_layer('logit').get_weights()]
    for a in range(0, len(x), chunk):
        seq = x[a:a + chunk].astype(np.float64)
        for n in C.GRU_NAMES:
            W, U, b = wts[n]
            u = U.shape[0]
            h = np.zeros((len(seq), u))
            outs = []
            for t in range(seq.shape[1]):
                xin = seq[:, t] @ W + b
                rec = h @ U[:, :2 * u]
                pre_zr = xin[:, :2 * u] + rec
                zr = sig(pre_zr)
                z, r = zr[:, :u], zr[:, u:]
                rh = r * h
                rec_h = rh @ U[:, 2 * u:]
                pre_h = xin[:, 2 * u:] + rec_h
                hh = np.tanh(pre_h)
                h = z * h + (1 - z) * hh
                for k, v in (('accum', xin), ('accum', rec), ('accum', pre_zr), ('accum', zr), ('accum', rec_h),
                             ('accum', pre_h), ('accum', hh), ('result', h), ('result', rh)):
                    upd(f'{n}.{k}', v)
                outs.append(h)
            seq = np.stack(outs, axis=1)
        upd('logit.accum', seq[:, -1] @ wd + bd if seq.ndim == 3 else seq @ wd + bd)
    for n in C.GRU_NAMES:
        W, U, b = wts[n]
        m[f'{n}.weight'], m[f'{n}.recurrent_weight'], m[f'{n}.bias'] = (float(np.abs(v).max()) for v in (W, U, b))
        m[f'{n}.table'] = 1.0 - 1e-6     # sigmoid and tanh table entries lie in (-1, 1)
    m['logit.weight'], m['logit.bias'] = float(np.abs(wd).max()), float(np.abs(bd).max())
    m['input'] = 1.0
    return m


t0 = time.time()
ranges = replay_ranges(model, d['x'])
ibits = {k: C.int_bits(v) for k, v in ranges.items()}
out['profile'] = {'rule': 'I = floor(log2 max|v|) + 2', 'max_abs': ranges, 'integer_bits': ibits, 's': round(time.time() - t0, 1)}
print('integer bits', json.dumps(ibits), flush=True)


def cfg_for(wbits, other_bits, table_size, default_tables):
    t = lambda w, k: f'ap_fixed<{w},{ibits[k]},AP_RND,AP_SAT>'
    cfg = C.base_config(model)
    cfg['LayerName']['syndrome_sequence']['Precision']['result'] = t(other_bits, 'input')
    for n in C.GRU_NAMES:
        p = cfg['LayerName'][n].setdefault('Precision', {})
        p['weight'], p['recurrent_weight'] = t(wbits, f'{n}.weight'), t(wbits, f'{n}.recurrent_weight')
        p['bias'] = p['recurrent_bias'] = t(wbits, f'{n}.bias')
        p['accum'], p['result'] = t(other_bits, f'{n}.accum'), t(other_bits, f'{n}.result')
        if default_tables:
            cfg['LayerName'][n]['table_size'] = 1024           # table_t stays at the hls4ml default ap_fixed<18,8>
        else:
            cfg['LayerName'][n]['table_size'] = table_size
            cfg['LayerName'][n]['table_t'] = t(other_bits, f'{n}.table')
    p = cfg['LayerName']['logit'].setdefault('Precision', {})
    p['weight'], p['bias'] = t(wbits, 'logit.weight'), t(wbits, 'logit.bias')
    p['accum'] = p['result'] = t(other_bits, 'logit.accum')
    cfg['LayerName']['logit_linear']['Precision']['result'] = t(other_bits, 'logit.accum')
    return cfg


CONFIGS = [(f'P16_t{args.table_size}', 16, 16, False), (f'P8_t{args.table_size}', 8, 16, False),
           ('P16_tdefault', 16, 16, True), ('P8_tdefault', 8, 16, True)]
out['configs'] = {}
for name, wb, ob, default_tables in CONFIGS:
    cfg = cfg_for(wb, ob, args.table_size, default_tables)
    prj = os.path.join(C.SCRATCH, f'eaf_{name}')
    shutil.rmtree(prj, ignore_errors=True)
    hm = C.convert(model, cfg, prj)
    hm.compile()
    t1 = time.time()
    y = C.csim_predict(hm, x, workers=args.workers)
    rec = {'weight_bits': wb, 'other_bits': ob, 'default_tables': default_tables,
           'csim_s': round(time.time() - t1, 1), 'stats_vs_fp32_keras': C.paired(kl > 0, y > 0, truth),
           'max_abs_logit_diff': float(np.abs(y - kl).max()), 'mean_abs_logit_diff': float(np.abs(y - kl).mean()),
           'mwpm_pL_same_shots': float((d['mwpm_pred'].astype(bool) != truth).mean())}
    rec['pL_hls_over_mwpm'] = rec['stats_vs_fp32_keras']['pL_hls'] / rec['mwpm_pL_same_shots']
    np.save(os.path.join(C.SCRATCH, f'csim_logits_{name}.npy'), y.astype(np.float32))
    # 100-shot testbench in the hls4ml format: one shot per line, values separated by spaces
    tb = os.path.join(prj, 'tb_data')
    os.makedirs(tb, exist_ok=True)
    np.savetxt(os.path.join(tb, 'tb_input_features.dat'), x[:100].reshape(100, -1), fmt='%g')
    np.savetxt(os.path.join(tb, 'tb_output_predictions.dat'), y[:100].reshape(100, 1), fmt='%.10g')
    np.savetxt(os.path.join(tb, 'keras_fp32_logits_100.dat'), kl[:100].reshape(100, 1), fmt='%.10g')
    C.dump(cfg, os.path.join(prj, 'hls4ml_config_used.json'))
    rec['project'] = prj
    rec['types'] = {l.split()[-1].rstrip(';'): ' '.join(l.split()[1:-1])
                    for l in open(os.path.join(prj, 'firmware', 'defines.h')) if l.startswith('typedef')}
    out['configs'][name] = rec
    print(name, json.dumps({k: rec[k] for k in ('stats_vs_fp32_keras', 'max_abs_logit_diff', 'pL_hls_over_mwpm', 'csim_s')}), flush=True)
    C.dump(out, os.path.join(C.HERE, 'results', 'step_b.json'))
