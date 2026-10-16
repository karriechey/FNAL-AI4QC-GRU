# Created: 2026-10-03 | Last updated: 2026-10-03
"""Step 3: post-training quantization (PTQ) bit-width sweep of the u70 GRU on hls4ml main
2ed80fb9, C simulation only (no synthesis).

1. Profile. Ranges on the evaluation block (all 1,800,000 shots):
   - hls4ml.model.profiling.weights_keras / activations_keras (layer weights and layer outputs);
   - the GRU's internal quantities, which hls4ml profiling cannot see, from a float64 numpy replay
     of the Keras recurrence (reset_after=False), every timestep: input product x.W + b,
     recurrent products h.U + b_r and (r*h).U_h + b_r, gate pre-activations, gate outputs,
     candidate, state h and r*h, Dense output.
   The integer bits of each hls4ml type are the smallest I with 2^(I-1) > max |value| (signed
   ap_fixed, one sign bit), at least 1. Saved to results/step3_profiled_ranges.json.

2. Sweep total width W in (24, 18, 16, 12, 10, 8). Every type of every layer gets
   ap_fixed<W, I_type>, I_type from step 1 (granularity='name'). If I_type >= W the type keeps
   one fractional bit and its integer part is cut ("capped"): those values can overflow.
   Two rounding / overflow modes, both reported:
     trn_wrap  hls4ml default (truncate, wrap around on overflow)
     rnd_sat   round to nearest, saturate on overflow
   For each width and mode: C simulation on all 1,800,000 shots, logical error rate against
   FP32 Keras on identical shots, exact McNemar test (b = Keras right and HLS wrong,
   c = HLS right and Keras wrong), and a per-layer trace (hls4ml Trace, 10,000 shots) of the GRU
   output and the logit against Keras.

3. "Which type limits first": at the largest width that fails (McNemar p < 0.05), each type
   group is put back to ap_fixed<32,16> one at a time (10,000-shot trace + 200,000-shot decision
   flips against Keras); the group whose restoration removes most of the error is the limiting one.

Not synthesized. Strategy Resource, ReuseFactor 35 (the arithmetic does not depend on the reuse
factor: Step 1 logits are bit-identical for RF 35, 140 and the kernel-fixed RF 70).

Usage (grace1):  ~/hls4ml_main_env/bin/python step3_ptq_sweep.py [--workers 48]
Writes: results/step3_profiled_ranges.json, results/step3_ptq_<mode>_w<W>.json,
        results/step3_ptq_table.csv, results/step3_ptq_table.md, results/step3_ablation_*.json
"""
import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np

import u70_common as U

ap = argparse.ArgumentParser()
ap.add_argument('--workers', type=int, default=48)
ap.add_argument('--widths', default='24,18,16,12,10,8')
ap.add_argument('--modes', default='trn_wrap,rnd_sat')
args = ap.parse_args()
WIDTHS = [int(w) for w in args.widths.split(',')]
MODES = args.modes.split(',')
RF = 35
TRACE_N = 10000
ABLATION_N = 200000
RES = os.path.join(U.HERE, 'results')
os.makedirs(RES, exist_ok=True)

env = U.environment_record()
model = U.build_model()
d = U.load_eval()
x, truth = d['x'], d['truth']
y_keras = model.predict(x, batch_size=20000, verbose=0).reshape(-1)
assert ((y_keras > 0) == (d['keras_logit_eaf'] > 0)).all()
pk = y_keras > 0


def int_bits(maxabs):
    """Smallest signed integer width I (incl. sign bit) with 2^(I-1) > maxabs, at least 1."""
    return max(1, math.floor(math.log2(maxabs)) + 2) if maxabs > 0 else 1


# ---- 1. profile ---------------------------------------------------------------------------
def gru_internal_ranges(model, x, chunk=100000):
    """float64 replay of the Keras GRU (reset_after=False, gate order z, r, h) on every
    timestep; returns max |value| of each quantity the hls4ml kernel stores."""
    W, Uw, b = [w.astype(np.float64) for w in model.get_layer('gru').get_weights()]
    wd, bd = [w.astype(np.float64) for w in model.get_layer('logit').get_weights()]
    n = Uw.shape[0]
    sig = lambda v: 1.0 / (1.0 + np.exp(-v))
    m = {}
    upd = lambda k, v: m.__setitem__(k, max(m.get(k, 0.0), float(np.abs(v).max())))
    for a in range(0, len(x), chunk):
        xb = x[a:a + chunk].astype(np.float64)
        h = np.zeros((len(xb), n))
        for t in range(xb.shape[1]):
            xin = xb[:, t] @ W + b                      # tmpres (input product + bias)
            hU = h @ Uw                                 # tmpres_state_zr (all 3n; h part unused)
            upd('input_product_plus_bias', xin)
            upd('recurrent_product_zr', hU[:, :2 * n])
            upd('recurrent_product_unused_h_part', hU[:, 2 * n:])
            pre_zr = xin[:, :2 * n] + hU[:, :2 * n]
            upd('preactivation_zr', pre_zr)
            zr = sig(pre_zr)
            z, r = zr[:, :n], zr[:, n:]
            rh = r * h
            upd('reset_times_h', rh)
            rU = rh @ Uw[:, 2 * n:]                     # tmpres_state_h
            upd('recurrent_product_candidate', rU)
            pre_h = xin[:, 2 * n:] + rU
            upd('preactivation_candidate', pre_h)
            hh = np.tanh(pre_h)
            upd('gate_outputs', zr); upd('candidate', hh)
            h = z * h + (1 - z) * hh
            upd('state_h', h)
        upd('logit', h @ wd + bd)
    return m


t0 = time.time()
from hls4ml.model import profiling as P
clean = lambda rows: [{k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in row.items()}
                      for row in rows]
prof_hls4ml = {}
# hls4ml main 2ed80fb9: weights_keras raises IndexError on a GRU layer (three weight arrays, its
# label list has two entries); the error is recorded and the weight ranges come from the arrays.
for key, fn in (('weights_keras', lambda: P.weights_keras(model, fmt='summary', plot='boxplot')),
                ('activations_keras_first200000', lambda: P.activations_keras(model, x[:200000], fmt='summary', plot='boxplot'))):
    try:
        prof_hls4ml[key] = clean(fn())
    except Exception as e:
        prof_hls4ml[key] = f'failed: {type(e).__name__}: {e}'
        print(f'hls4ml profiling {key} failed: {type(e).__name__}: {e}', flush=True)
internal = gru_internal_ranges(model, x)
gw, gu, gb = model.get_layer('gru').get_weights()
dw, db = model.get_layer('logit').get_weights()
# The activation tables hold sigmoid / tanh on the input grid [-8, 8) (hls4ml default, 1024 entries).
grid = np.linspace(-8, 8, 1024, endpoint=False)
table_max = float(max(np.abs(1 / (1 + np.exp(-grid))).max(), np.abs(np.tanh(grid)).max()))
ranges = {
    'input':                     1.0,  # syndrome bits 0/1
    'gru_weight':                float(np.abs(gw).max()),
    'gru_recurrent_weight':      float(np.abs(gu).max()),
    'gru_bias':                  float(np.abs(gb).max()),
    'gru_recurrent_bias':        float(np.abs(gb).max()),  # hls4ml splits the single Keras bias; take the same bound
    # accum_t holds the products, the pre-activations and the gate / candidate outputs
    'gru_accum':                 max(internal[k] for k in ('input_product_plus_bias', 'recurrent_product_zr',
                                                           'preactivation_zr', 'recurrent_product_candidate',
                                                           'preactivation_candidate', 'gate_outputs', 'candidate')),
    'gru_result':                max(internal['state_h'], internal['reset_times_h']),  # h_state and h_reset are res_T
    'gru_table':                 table_max,
    'logit_weight':              float(np.abs(dw).max()),
    'logit_bias':                float(np.abs(db).max()),
    'logit_accum':               internal['logit'],
    'logit_result':              internal['logit'],
}
ibits = {k: int_bits(v) for k, v in ranges.items()}
json.dump({'environment': env, 'shots': [U.EVAL_START, U.EVAL_STOP], 'rule': 'I = floor(log2(max|v|)) + 2, min 1',
           'max_abs': ranges, 'integer_bits': ibits, 'gru_internal_max_abs': internal, 'hls4ml_profiling': prof_hls4ml,
           'profile_s': round(time.time() - t0, 1)},
          open(os.path.join(RES, 'step3_profiled_ranges.json'), 'w'), indent=1, default=str)
print('integer bits:', json.dumps(ibits), '\nmax abs:', json.dumps(ranges), flush=True)

# ---- 2. sweep -------------------------------------------------------------------------------
MODE_SUFFIX = {'trn_wrap': '', 'rnd_sat': ',RND,SAT'}
# config key per type: (layer, precision key)
KEYS = {'input': ('syndrome_sequence', 'result'), 'gru_weight': ('gru', 'weight'),
        'gru_recurrent_weight': ('gru', 'recurrent_weight'), 'gru_bias': ('gru', 'bias'),
        'gru_recurrent_bias': ('gru', 'recurrent_bias'), 'gru_accum': ('gru', 'accum'),
        'gru_result': ('gru', 'result'), 'gru_table': ('gru', 'table'),
        'logit_weight': ('logit', 'weight'), 'logit_bias': ('logit', 'bias'),
        'logit_accum': ('logit', 'accum'), 'logit_result': ('logit', 'result')}
GROUPS = {'gru weights+biases': ['gru_weight', 'gru_recurrent_weight', 'gru_bias', 'gru_recurrent_bias'],
          'gru accum': ['gru_accum'], 'gru result (h)': ['gru_result'], 'gru activation table': ['gru_table'],
          'logit (all types)': ['logit_weight', 'logit_bias', 'logit_accum', 'logit_result'], 'input': ['input']}


def types_for(width, mode, keep_full=()):
    out = {}
    for k, I in ibits.items():
        if k in keep_full:
            out[k] = ('ap_fixed<32,16>', 32, 16, False)
            continue
        capped = I >= width
        i_used = width - 1 if capped else I
        out[k] = (f'ap_fixed<{width},{i_used}{MODE_SUFFIX[mode]}>', width, i_used, capped)
    return out


def make_cfg(types, trace=False):
    cfg = U.base_config(model, 'ap_fixed<32,16>', 'Resource', RF)
    for k, (prec, *_ ) in types.items():
        layer, key = KEYS[k]
        cfg['LayerName'][layer].setdefault('Precision', {})[key] = prec
    # linear activation after the Dense layer: same type as the Dense result
    cfg['LayerName']['logit_linear']['Precision']['result'] = types['logit_result'][0]
    if trace:
        for n in ('gru', 'logit'):
            cfg['LayerName'][n]['Trace'] = True
    return cfg


HM = None


def _work(bounds):
    a, b = bounds
    return a, U.csim_predict(HM, x[a:b])


def csim_all(hm, n):
    global HM
    HM = hm
    chunks = [(a, min(a + 2500, n)) for a in range(0, n, 2500)]
    y = np.empty(n)
    with mp.get_context('fork').Pool(min(args.workers, len(chunks))) as pool:
        for a, yy in pool.imap_unordered(_work, chunks):
            y[a:a + len(yy)] = yy
    return y


# Reference layer outputs from Keras sub-models. hls4ml's get_ymodel_keras cannot be used here:
# on main 2ed80fb9 it tries to assign GRU.activation, a read-only property in Keras 2.15.
import tensorflow as tf
keras_trace = {n: tf.keras.Model(model.input, model.get_layer(n).output).predict(x[:TRACE_N], batch_size=10000, verbose=0)
               for n in ('gru', 'logit')}


def trace_errors(hm):
    _, tr = hm.trace(np.ascontiguousarray(x[:TRACE_N]))
    out = {}
    for layer in ('gru', 'logit'):
        e = np.abs(np.asarray(tr[layer], float).reshape(TRACE_N, -1) - np.asarray(keras_trace[layer], float).reshape(TRACE_N, -1))
        out[layer] = {'max_abs_err': float(e.max()), 'mean_abs_err': float(e.mean()),
                      # an error above 0.5 on h in (-1, 1) or a logit is the signature of an overflow
                      'frac_shots_err_gt_0p5': float((e.max(axis=1) > 0.5).mean())}
    return out


def run_config(tag, types, n):
    prj = os.path.join(U.HERE, f'hls_step3_{tag}')
    hm = U.convert(model, make_cfg(types, trace=True), prj)
    hm.compile()
    t1 = time.time()
    y = csim_all(hm, n)
    st = U.paired_stats(pk[:n], y > 0, truth[:n])
    lo = (st['c_new_only_right'] - st['b_ref_only_right']) / n
    se = math.sqrt(st['n_discordant']) / n
    st['delta_p_L_new_minus_keras'] = st['p_L_new'] - st['p_L_ref']
    st['delta_p_L_95ci'] = [-lo - 1.96 * se, -lo + 1.96 * se]
    return y, st, trace_errors(hm), round(time.time() - t1, 1)


rows = []
for mode in MODES:
    for wdt in WIDTHS:
        types = types_for(wdt, mode)
        tag = f'{mode}_w{wdt}'
        y, st, tr, secs = run_config(tag, types, len(x))
        capped = [k for k, v in types.items() if v[3]]
        rec = {'label': f'PTQ width sweep (C simulation only), ap_fixed<{wdt},I>', 'mode': mode, 'width': wdt,
               'environment': env, 'types': {k: v[0] for k, v in types.items()}, 'capped_types': capped,
               'shots': [U.EVAL_START, U.EVAL_STOP], 'stats_vs_fp32_keras': st, 'trace_vs_keras_10k': tr,
               'csim_s': secs, 'indistinguishable_p_ge_0p05': st['p_exact'] >= 0.05}
        json.dump(rec, open(os.path.join(RES, f'step3_ptq_{tag}.json'), 'w'), indent=1, default=str)
        np.save(os.path.join(RES, f'step3_hls_logits_{tag}.npy'), y.astype(np.float32))
        rows.append({'mode': mode, 'width': wdt, 'p_L_keras': st['p_L_ref'], 'p_L_hls': st['p_L_new'],
                     'errors_keras': st['errors_ref'], 'errors_hls': st['errors_new'],
                     'b_keras_only_right': st['b_ref_only_right'], 'c_hls_only_right': st['c_new_only_right'],
                     'n_discordant': st['n_discordant'], 'p_exact': st['p_exact'],
                     'decision_match_rate': st['decision_match_rate'],
                     'delta_p_L_95ci_lo': st['delta_p_L_95ci'][0], 'delta_p_L_95ci_hi': st['delta_p_L_95ci'][1],
                     'indistinguishable': rec['indistinguishable_p_ge_0p05'],
                     'capped_types': ';'.join(capped) or 'none',
                     'gru_h_trace_max_err': tr['gru']['max_abs_err'], 'gru_h_trace_mean_err': tr['gru']['mean_abs_err'],
                     'logit_trace_max_err': tr['logit']['max_abs_err'], 'logit_trace_mean_err': tr['logit']['mean_abs_err'],
                     'frac_shots_overflow_signature_gru': tr['gru']['frac_shots_err_gt_0p5'],
                     'frac_shots_overflow_signature_logit': tr['logit']['frac_shots_err_gt_0p5']})
        print(json.dumps(rows[-1]), flush=True)

# ---- 3. limiting type at the largest failing width ----------------------------------------
ablations = {}
for mode in MODES:
    fails = [r['width'] for r in rows if r['mode'] == mode and not r['indistinguishable']]
    if not fails:
        continue
    wdt = max(fails)
    base_types = types_for(wdt, mode)
    out = {}
    for gname, keys in [('none restored', [])] + list(GROUPS.items()):
        types = types_for(wdt, mode, keep_full=keys)
        tag = f'abl_{mode}_w{wdt}_' + (keys[0] if keys else 'none')
        y, st, tr, _ = run_config(tag, types, ABLATION_N)
        out[gname] = {'decision_flips_vs_keras': int(((y > 0) != pk[:ABLATION_N]).sum()),
                      'b': st['b_ref_only_right'], 'c': st['c_new_only_right'], 'p_exact': st['p_exact'],
                      'gru_h_trace_mean_err': tr['gru']['mean_abs_err'], 'logit_trace_mean_err': tr['logit']['mean_abs_err']}
        print(mode, wdt, gname, json.dumps(out[gname]), flush=True)
    ablations[f'{mode}_w{wdt}'] = out
    json.dump({'mode': mode, 'width': wdt, 'n': ABLATION_N, 'restored_to': 'ap_fixed<32,16>', 'groups': GROUPS,
               'results': out, 'environment': env},
              open(os.path.join(RES, f'step3_ablation_{mode}_w{wdt}.json'), 'w'), indent=1, default=str)

# ---- table ----------------------------------------------------------------------------------
with open(os.path.join(RES, 'step3_ptq_table.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['hls4ml_commit'] + list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow({'hls4ml_commit': env['hls4ml_commit'], **r})
md = ['# u70 GRU PTQ width sweep (C simulation only, not synthesized)', '',
      f"hls4ml `{env['hls4ml_commit']}`; {len(x):,} shots of the FORMAL evaluation block [15.2M, 17.0M) "
      '(EAF pool); reference = FP32 Keras on the same shots. Integer bits per type from profiling '
      '(results/step3_profiled_ranges.json); b = Keras right & HLS wrong, c = HLS right & Keras wrong.', '',
      '| mode | W | p_L HLS | p_L Keras | b | c | n_disc | McNemar p | ΔLER 95% CI | same as FP32? | capped types | h trace mean err |',
      '|---|---|---|---|---|---|---|---|---|---|---|---|']
for r in rows:
    md.append(f"| {r['mode']} | {r['width']} | {r['p_L_hls']:.6f} | {r['p_L_keras']:.6f} | {r['b_keras_only_right']} | "
              f"{r['c_hls_only_right']} | {r['n_discordant']} | {r['p_exact']:.3g} | "
              f"[{r['delta_p_L_95ci_lo']:+.2e}, {r['delta_p_L_95ci_hi']:+.2e}] | {'yes' if r['indistinguishable'] else 'no'} | "
              f"{r['capped_types']} | {r['gru_h_trace_mean_err']:.2e} |")
for mode in MODES:
    ok = [r['width'] for r in rows if r['mode'] == mode]
    # smallest width W such that W and every larger swept width are indistinguishable from FP32
    good = None
    for wdt in sorted(ok, reverse=True):
        if next(r for r in rows if r['mode'] == mode and r['width'] == wdt)['indistinguishable']:
            good = wdt
        else:
            break
    md.append(f'\n{mode}: smallest width indistinguishable from FP32 (it and every wider swept width p >= 0.05): {good}')
for k, v in ablations.items():
    md.append(f'\nLimiting type at {k} ({ABLATION_N:,} shots, one group restored to ap_fixed<32,16> at a time; '
              'decision flips vs Keras):')
    for g, r in v.items():
        md.append(f'- {g}: {r["decision_flips_vs_keras"]} flips, h trace mean err {r["gru_h_trace_mean_err"]:.2e}')
md.append('\nInteger bits used: ' + ', '.join(f'{k} {v}' for k, v in ibits.items()))
open(os.path.join(RES, 'step3_ptq_table.md'), 'w').write('\n'.join(md) + '\n')
print('\n'.join(md))
