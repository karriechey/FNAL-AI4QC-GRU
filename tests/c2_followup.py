# Created: 2026-09-26
# Last modified: 2026-09-26

"""C2 follow-up: why does the trained reset_after=False u140 diverge in hls4ml?

1 pre-activation ranges from a numpy GRU, against the ap_fixed<32,16> limit
2 hls4ml vs Keras with the trained checkpoint
3 the same with every GRU bias zeroed, to isolate the bias handling
Default and 65,536-entry tables; first 500 shots of the d=5 eval block.

python tests/c2_followup.py --out <dir> --u70-weights <path to gkd20m_u70 .weights.h5>
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
D, N_SHOTS, PRECISION = 5, 500, 'ap_fixed<32,16>'


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def float_gru_ranges(model, x):
    # numpy GRU (reset_after=False); max |.| of every pre-activation
    W, U, b = [np.asarray(v, np.float64) for v in model.get_layer('gru').get_weights()]
    u = U.shape[0]
    h = np.zeros((x.shape[0], u))
    mx = dict(xW_plus_b=0.0, hU_zr=0.0, candidate_preact=0.0, h=0.0)
    for t in range(x.shape[1]):
        xw = x[:, t, :] @ W + b
        hu = h @ U[:, :2 * u]
        z = sigmoid(xw[:, :u] + hu[:, :u])
        r = sigmoid(xw[:, u:2 * u] + hu[:, u:])
        cand_pre = xw[:, 2 * u:] + (r * h) @ U[:, 2 * u:]
        h = z * h + (1 - z) * np.tanh(cand_pre)
        mx['xW_plus_b'] = max(mx['xW_plus_b'], float(np.abs(xw).max()))
        mx['hU_zr'] = max(mx['hU_zr'], float(np.abs(hu).max()))
        mx['candidate_preact'] = max(mx['candidate_preact'], float(np.abs(cand_pre).max()))
        mx['h'] = max(mx['h'], float(np.abs(h).max()))
    Wd, bd = model.get_layer('logit').get_weights()
    logit = h @ Wd + bd
    keras_logit = model.predict(x.astype(np.float32), verbose=0)
    mx['logit'] = float(np.abs(logit).max())
    mx['numpy_vs_keras_max_diff'] = float(np.abs(logit - keras_logit).max())
    return dict(bias_abs_max=float(np.abs(b).max()), bias_abs_mean=float(np.abs(b).mean()),
                **mx)


def compare(tf, hls4ml, model, x, outdir, name, table):
    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name',
                                               default_precision=PRECISION, backend='Vivado')
    if table:
        cfg['LayerName']['gru']['TableSize'] = 65536
        cfg['LayerName']['gru']['Precision']['table'] = PRECISION
    hm = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, output_dir=os.path.join(outdir, f'hls_{name}'),
        backend='Vivado', io_type='io_parallel')
    hm.compile()
    yk = model.predict(x, verbose=0).reshape(-1)
    yh = np.asarray(hm.predict(x)).reshape(-1)
    d = np.abs(yh - yk)
    return dict(max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()),
                sign_agreement=float(np.mean(np.sign(yh) == np.sign(yk))),
                keras_logit_std=float(yk.std()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--u70-weights', default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf
    import hls4ml
    import partitions as P
    from StudentModels import to_sequence, build_gru_student

    e0 = P.block('eval')[0]
    det = np.load(P.pool_path(D, 'formal'))['det_evts'][e0:e0 + N_SHOTS]
    x = np.ascontiguousarray(to_sequence(det, D, D, P.load()['p']), dtype=np.float32)
    paths = P._load('paths.yaml')
    ckpts = {'u140_run_of_record': (140, os.path.join(
        os.path.expanduser(paths['results_root']), paths['runs_of_record']['d5_u140_10M'],
        'seed0', 'ckpt', 'gru_d5_p004_u140_hard_seed0_ntr10000000_b10000.best.weights.h5'))}
    if a.u70_weights:
        ckpts['u70_gkd20m_hard_seed0'] = (70, a.u70_weights)

    rep = dict(hls4ml=hls4ml.__version__, tensorflow=tf.__version__, cases={})
    for name, (units, path) in ckpts.items():
        m = build_gru_student(d=D, rounds=D, units=units, p=P.load()['p'])
        m.load_weights(path)
        r = dict(weights=path, ranges=float_gru_ranges(m, x.astype(np.float64)))
        for table in (False, True):
            tag = 'tables65536' if table else 'default_tables'
            r[f'trained_{tag}'] = compare(tf, hls4ml, m, x, a.out, f'{name}_trained_{tag}', table)
            ws = m.get_layer('gru').get_weights()
            m0 = build_gru_student(d=D, rounds=D, units=units, p=P.load()['p'])
            m0.set_weights(m.get_weights())
            m0.get_layer('gru').set_weights([ws[0], ws[1], np.zeros_like(ws[2])])
            r[f'zero_bias_{tag}'] = compare(tf, hls4ml, m0, x, a.out,
                                            f'{name}_zero_bias_{tag}', table)
        rep['cases'][name] = r
        print(f'[C2f] {name}: ' + json.dumps(r), flush=True)
        with open(os.path.join(a.out, 'c2_followup.json'), 'w') as fh:
            json.dump(rep, fh, indent=1)


if __name__ == '__main__':
    main()
