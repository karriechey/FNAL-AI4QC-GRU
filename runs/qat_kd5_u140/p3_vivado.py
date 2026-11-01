# Created: 2026-10-28 | Last updated: 2026-10-28
# Vivado-backend projects of the Phase 3 configurations, for C synthesis with Vivado HLS 2020.1 on EAF.
# --rf other than 70 makes <config>_rf<RF>_vivado; its C-sim is still compared with the verified RF 70 Vitis C-sim.
# Same on-grid Keras model and hls4ml config as p3_hls.py (Vitis backend); a config is exported only if
#   1. its firmware sources equal the Vitis project's (build scripts aside), and
#   2. its C-sim logits on the first --n eval-block shots equal the saved Vitis C-sim logits (max |diff| 0.0).
#   bash hpy.sh p3_vivado.py --n 100000 qat:w12_s6_seed0 qat:w10_s8_seed0 qat:w12_s8_seed0 qat:w16_s6_seed0
import argparse
import filecmp
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcommon as Q
import hls_util as H
import numpy as np
import fq_gru as F

ap = argparse.ArgumentParser()
ap.add_argument('configs', nargs='+')
ap.add_argument('--n', type=int, default=100_000)
ap.add_argument('--workers', type=int, default=48)
ap.add_argument('--rf', type=int, default=70)
ap.add_argument('--out', default=os.path.join(Q.HERE, 'hls_export'))
args = ap.parse_args()

x = Q.load_eval(args.n)['x'].astype(np.float32)


def load(config):
    name = config.split(':')[1]
    res = json.load(open(os.path.join(Q.HERE, 'qat_runs', name, 'result.json')))
    spec = {k: tuple(v) for k, v in res['spec'].items()}
    fq = F.build_fq(spec)
    fq.load_weights(os.path.join(Q.HERE, 'qat_runs', name, 'best.weights.h5'))
    m = Q.build_fp32()
    W, U, b = fq.get_layer('gru').get_weights()
    wd, bd = fq.get_layer('logit').get_weights()
    qz = lambda v, g: F.quantize_np(v, *spec[g]).astype(np.float32)
    m.get_layer('gru').set_weights([qz(W, 'gru_weight'), qz(U, 'gru_recurrent_weight'), qz(b, 'gru_bias')])
    m.get_layer('logit').set_weights([qz(wd, 'logit_weight'), qz(bd, 'logit_bias')])
    return name, spec, m


def firmware_diff(a, b):
    # files under firmware/ that differ or exist in one project only
    out = []
    for root, _, files in os.walk(os.path.join(a, 'firmware')):
        for f in files:
            pa = os.path.join(root, f)
            pb = os.path.join(b, os.path.relpath(pa, a))
            if f.endswith('.so'):
                continue
            if not os.path.exists(pb) or not filecmp.cmp(pa, pb, shallow=False):
                out.append(os.path.relpath(pa, a))
    for root, _, files in os.walk(os.path.join(b, 'firmware')):
        for f in files:
            pb = os.path.join(root, f)
            if not f.endswith('.so') and not os.path.exists(os.path.join(a, os.path.relpath(pb, b))):
                out.append(os.path.relpath(pb, b) + ' (Vitis only)')
    return out


import hls4ml
summary = {}
for config in args.configs:
    name, spec, model = load(config)
    tag = f'qat_{name}'
    cfg = Q.hls_config(model, spec, reuse_factor=args.rf)
    rec = {'config': config, 'n': args.n, 'reuse_factor': args.rf}
    prj = {}
    for backend in ('Vivado', 'Vitis'):
        prj[backend] = os.path.join(Q.SCRATCH, 'hls', f'p3v_{tag}_rf{args.rf}_{backend.lower()}')
        hm = hls4ml.converters.convert_from_keras_model(
            model, hls_config=json.loads(json.dumps(cfg)), output_dir=prj[backend], project_name='myproject',
            backend=backend, io_type=Q.IO_TYPE, part=Q.PART, clock_period=Q.CLOCK_NS)
        hm.write()
        if backend == 'Vivado':
            hm_vivado = hm
    rec['firmware_files_differing_from_vitis'] = firmware_diff(prj['Vivado'], prj['Vitis'])
    hm_vivado.compile()
    y = H.csim_predict(hm_vivado, x, workers=args.workers)
    vitis = np.load(os.path.join(Q.SCRATCH, 'logits', f'hls_{tag}.npy'))[:args.n].astype(np.float64)
    rec['max_abs_diff_vs_vitis_csim'] = float(np.abs(y.astype(np.float32) - vitis).max())
    rec['decision_agreement_vs_vitis_csim'] = float(((y > 0) == (vitis > 0)).mean())
    gen = H.generated_types(prj['Vivado'])
    rec['types'] = gen['typedefs']
    rec['generated_schedule'] = {k: gen[k] for k in ('n_unit_par', 'n_cycles', 'reuse_factor')}
    if rec['max_abs_diff_vs_vitis_csim'] != 0.0:
        rec['status'] = 'stopped: Vivado C-sim differs from the verified Vitis C-sim'
    else:
        out = os.path.join(args.out, f'{tag}_vivado' if args.rf == 70 else f'{tag}_rf{args.rf}_vivado')
        shutil.rmtree(out, ignore_errors=True)
        shutil.copytree(prj['Vivado'], os.path.join(out, 'project'), ignore=shutil.ignore_patterns('*.so', '*.o'))
        model.save_weights(os.path.join(out, 'keras_on_grid.weights.h5'))
        open(os.path.join(out, 'keras_model.json'), 'w').write(model.to_json())
        Q.dump(cfg, os.path.join(out, 'hls4ml_config.json'))
        Q.dump({'spec_W_I': spec, 'rounding': 'AP_RND', 'overflow': 'AP_SAT', 'input': 'ap_uint<1>',
                'table_size': Q.TABLE_SIZE}, os.path.join(out, 'precisions.json'))
        import platform
        import tensorflow as tf
        Q.dump({**Q.hls4ml_record(), 'tensorflow': tf.__version__, 'numpy': np.__version__,
                'python': platform.python_version(), 'machine': platform.machine()}, os.path.join(out, 'versions.json'))
        Q.dump({'part': Q.PART, 'clock_period_ns': Q.CLOCK_NS, 'io_type': Q.IO_TYPE, 'backend': 'Vivado',
                'strategy': 'Resource', 'reuse_factor': args.rf, 'static_gru': True, 'hls_tool': 'Vivado HLS 2020.1',
                'csim_check': {k: rec[k] for k in ('n', 'max_abs_diff_vs_vitis_csim',
                                                   'firmware_files_differing_from_vitis')}},
               os.path.join(out, 'target.json'))
        for f in ('run_synth.sh', 'collect_csynth.py'):
            shutil.copy(os.path.join(Q.HERE, f), out)
        rec['status'] = 'exported'
        rec['export'] = out
    summary[tag] = rec
    print(json.dumps({k: v for k, v in rec.items() if k != 'types'}), flush=True)
Q.dump(summary, os.path.join(Q.HERE, 'results', 'p3_vivado_check.json' if args.rf == 70 else f'p3_vivado_check_rf{args.rf}.json'))
