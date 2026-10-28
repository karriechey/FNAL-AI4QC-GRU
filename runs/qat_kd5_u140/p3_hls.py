# Created: 2026-10-28 | Last updated: 2026-10-28
# Phase 3: hls4ml (Vitis backend, io_parallel, Resource) for selected configurations.
# Weights are put on their ap_fixed grid in Keras first, then converted with the same per-layer
# precisions; C-sim on the first --n eval-block shots, compared with the emulator (the quantized
# Keras model) and scored against FP32 and MWPM on the same shots. Below 99.9% agreement:
# widen accum_t, then table_size / table_t, then the state, one at a time (on 100k shots).
# Each configuration is exported to hls_export/<config>/ for C synthesis on x86 (run_synth.sh).
#   bash hpy.sh p3_hls.py --n 1000000 qat:w8_s8_seed0 ptq:w12_s16 ...
import argparse
import json
import os
import platform
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcommon as Q
import hls_util as H
import numpy as np
import fq_gru as F

ap = argparse.ArgumentParser()
ap.add_argument('configs', nargs='+', help='qat:<run name> or ptq:<spec tag>')
ap.add_argument('--n', type=int, default=1_000_000)
ap.add_argument('--workers', type=int, default=48)
ap.add_argument('--rf', type=int, default=70)
args = ap.parse_args()

EXPORT = os.path.join(Q.HERE, 'hls_export')
d_all = Q.load_eval(args.n)
x = d_all['x'].astype(np.float32)
fp32 = Q.build_fp32()


def load(config):
    kind, name = config.split(':')
    if kind == 'qat':
        res = json.load(open(os.path.join(Q.HERE, 'qat_runs', name, 'result.json')))
        spec = {k: tuple(v) for k, v in res['spec'].items()}
        fq = F.build_fq(spec)
        fq.load_weights(os.path.join(Q.HERE, 'qat_runs', name, 'best.weights.h5'))
    else:
        spec = {k: tuple(v) for k, v in json.load(open(os.path.join(Q.HERE, 'specs', f'{name}.json'))).items()}
        fq = F.build_fq(spec)
        F.copy_weights(fp32, fq)
    return kind, name, spec, fq


def on_grid_keras(fq, spec):
    m = Q.build_fp32()
    W, U, b = fq.get_layer('gru').get_weights()
    wd, bd = fq.get_layer('logit').get_weights()
    qz = lambda v, g: F.quantize_np(v, *spec[g]).astype(np.float32)
    m.get_layer('gru').set_weights([qz(W, 'gru_weight'), qz(U, 'gru_recurrent_weight'), qz(b, 'gru_bias')])
    m.get_layer('logit').set_weights([qz(wd, 'logit_weight'), qz(bd, 'logit_bias')])
    return m


def convert(model, spec, prj, table_size=Q.TABLE_SIZE):
    import hls4ml
    cfg = Q.hls_config(model, spec, reuse_factor=args.rf, table_size=table_size)
    hm = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, output_dir=prj, project_name='myproject', backend='Vitis',
        io_type=Q.IO_TYPE, part=Q.PART, clock_period=Q.CLOCK_NS)
    hm.write()
    return hm, cfg


def compile_with_fallback(hm, rec):
    try:
        hm.compile()
        return True
    except Exception as e:
        rec['compile_error'] = f'{type(e).__name__}: {e}'
    # one fix: the project's own build script with g++ -O2 and no -march flags
    try:
        os.environ['CXX'] = 'g++'
        hm.compile()
        rec['compile_fix'] = 'recompiled with CXX=g++'
        return True
    except Exception as e:
        rec['compile_error_after_fix'] = f'{type(e).__name__}: {e}'
        return False


def versions():
    import tensorflow as tf
    import tf_keras
    import qkeras
    return {**Q.hls4ml_record(), 'tensorflow': tf.__version__, 'tf_keras': tf_keras.__version__,
            'qkeras': getattr(qkeras, '__version__', 'unknown'), 'numpy': np.__version__,
            'python': platform.python_version(), 'machine': platform.machine(), 'host': platform.node()}


def export(tag, prj, cfg, model, spec, rec):
    out = os.path.join(EXPORT, tag)
    if os.path.exists(out):
        out = out + time.strftime('_%Y%m%dT%H%M%SZ', time.gmtime())
    os.makedirs(out)
    shutil.copytree(prj, os.path.join(out, 'project'),
                    ignore=shutil.ignore_patterns('*.so', '*.o', 'tb_data'))
    model.save_weights(os.path.join(out, 'keras_on_grid.weights.h5'))
    open(os.path.join(out, 'keras_model.json'), 'w').write(model.to_json())
    Q.dump(cfg, os.path.join(out, 'hls4ml_config.json'))
    Q.dump({'spec_W_I': spec, 'rounding': 'AP_RND', 'overflow': 'AP_SAT', 'input': 'ap_uint<1>',
            'table_size': rec.get('table_size', Q.TABLE_SIZE)}, os.path.join(out, 'precisions.json'))
    Q.dump(versions(), os.path.join(out, 'versions.json'))
    Q.dump({'part': Q.PART, 'clock_period_ns': Q.CLOCK_NS, 'io_type': Q.IO_TYPE, 'backend': 'Vitis',
            'strategy': 'Resource', 'reuse_factor': args.rf, 'static_gru': True}, os.path.join(out, 'target.json'))
    shutil.copy(os.path.join(Q.HERE, 'run_synth.sh'), out)
    shutil.copy(os.path.join(Q.HERE, 'collect_csynth.py'), out)
    return out


summary = []
for config in args.configs:
    kind, name, spec, fq = load(config)
    tag = f'{kind}_{name}'
    rec = {'config': config, 'n': args.n, 'spec': spec}
    y_fq = F.predict_logits(fq, x)
    model = on_grid_keras(fq, spec)
    prj = os.path.join(Q.SCRATCH, 'hls', f'p3_{tag}')
    hm, cfg = convert(model, spec, prj)
    if not compile_with_fallback(hm, rec):
        rec['status'] = 'compile failed on aarch64; C-sim must be rerun on x86'
        rec['export'] = export(tag, prj, cfg, model, spec, rec)
        summary.append(rec)
        print(json.dumps(rec, default=str), flush=True)
        continue
    rec['types'] = H.generated_types(prj)
    n_probe = min(100_000, args.n)
    y = H.csim_predict(hm, x[:n_probe], workers=args.workers)
    rec['probe_agreement'] = float(((y > 0) == (y_fq[:n_probe] > 0)).mean())
    fixes = []
    if rec['probe_agreement'] < 0.999:
        wider_state = {g: (spec[g][0] + 2, spec[g][1]) for g in ('gru_state',)}
        for label, sp, ts in (('accum_t 24 -> 32 bits', {**spec, 'gru_accum': (32, spec['gru_accum'][1])}, Q.TABLE_SIZE),
                              ('table_size 4096, table_t +4 bits', {**spec, 'gru_table': (spec['gru_table'][0] + 4, spec['gru_table'][1])}, 4096),
                              ('state +2 bits', {**spec, **wider_state}, Q.TABLE_SIZE)):
            p2 = prj + '_fix'
            hm2, cfg2 = convert(model, sp, p2, table_size=ts)
            hm2.compile()
            y2 = H.csim_predict(hm2, x[:n_probe], workers=args.workers)
            a2 = float(((y2 > 0) == (y_fq[:n_probe] > 0)).mean())
            fixes.append({'change': label, 'probe_agreement': a2})
            if a2 >= 0.999:
                hm, cfg, prj, spec = hm2, cfg2, p2, sp
                rec['table_size'] = ts
                break
    rec['fixes_tried'] = fixes
    t0 = time.time()
    y = H.csim_predict(hm, x, workers=args.workers)
    rec['csim_s'] = round(time.time() - t0, 1)
    np.save(os.path.join(Q.SCRATCH, 'logits', f'hls_{tag}.npy'), y.astype(np.float32))
    rec['agreement_vs_quantized_keras'] = float(((y > 0) == (y_fq > 0)).mean())
    rec['max_abs_logit_diff'] = float(np.abs(y - y_fq).max())
    rec['hls'] = Q.score(y > 0, d_all)
    rec['emulator'] = Q.score(y_fq > 0, d_all)
    rec['status'] = 'ok' if rec['agreement_vs_quantized_keras'] >= 0.999 else 'agreement below 99.9%'
    rec['export'] = export(tag, prj, cfg, model, spec, rec)
    Q.dump(rec, os.path.join(Q.HERE, 'results', f'p3_{tag}.json'))
    summary.append(rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ('types',)}, default=str), flush=True)
