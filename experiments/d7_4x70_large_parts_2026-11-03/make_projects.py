# Created: 2026-11-03 | Last updated: 2026-11-03
# Writes the EAF projects from the d7_4x70 configs and checks each C-sim against the RF 70 stock logits
#   bash hpy.sh make_projects.py --n 100000
#   bash hpy.sh make_projects.py --only vivado_P8_t4096_rf70 --n 0 (0 = full eval block)
import argparse
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'd7_4x70_hls_2026-10-29'))
import d7_common as C
import hls4ml
import numpy as np

OUT = '/tmp/kchey_qat/d7_4x70_parts'
PARTS = {'vivado': 'xcvu13p-flga2577-2-e', 'vitis': 'xcvp1902-vsva6865-2MP-e-S'}
PROJECTS = [('vivado', 'P8_t4096', 70), ('vivado', 'P8_t4096', 35), ('vivado', 'P8_t4096', 14),
            ('vitis', 'P8_t4096', 70), ('vitis', 'P8_t4096', 35), ('vitis', 'P8_t4096', 20),
            ('vivado', 'P16_tdefault', 35), ('vivado', 'P16_tdefault', 14), ('vitis', 'P16_tdefault', 20)]
# variable loop bound in gru_stack: HLS cannot unroll it, so GRU latency is unknown and h_state gets muxed
STOCK_LOOP = re.compile(r'for \(int i = CONFIG_T::n_state \* iloop, j = 0; i < \(CONFIG_T::n_state \* \(iloop \+ 1\)\); '
                        r'i\+\+, j\+\+\) \{(\s+#pragma HLS UNROLL\s+)res\[i\] = h_state\[j\];')
FIXED_LOOP = r'for (int j = 0; j < CONFIG_T::n_state; j++) {\1res[j + CONFIG_T::n_state * iloop] = h_state[j];'


def patch_output_loop(prj):
    path = os.path.join(prj, 'firmware', 'nnet_utils', 'nnet_recurrent.h')
    text = open(path).read()
    text, n = STOCK_LOOP.subn(FIXED_LOOP, text)
    assert n >= 2, n
    open(path, 'w').write(text)
    return n


def config_at_rf(name, rf):
    cfg = json.load(open(os.path.join(C.SCRATCH, f'eaf_{name}', 'hls4ml_config_used.json')))
    cfg['Model']['ReuseFactor'] = rf
    for layer in C.GRU_NAMES:
        cfg['LayerName'][layer]['ReuseFactor'] = rf
    # Dense RF below 35 chains 4+ saturating adds per cycle, which missed 5 ns on kd5_u140
    cfg['LayerName']['logit']['ReuseFactor'] = max(rf, 35)
    return cfg


ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=100_000)
ap.add_argument('--only', default=None)
ap.add_argument('--workers', type=int, default=30)
args = ap.parse_args()

n = args.n or None
d = C.load_eval(n)
x, truth = d['x'].astype(np.float32), d['truth'].astype(bool)
kl = d['keras_logit'].astype(np.float64)
model = C.build_model()
rows = {}
for backend, name, rf in PROJECTS:
    tag = f'{backend}_{name}_rf{rf}'
    if args.only and tag != args.only:
        continue
    prj = os.path.join(OUT, tag if n else tag + '_full')
    shutil.rmtree(prj, ignore_errors=True)
    cfg = config_at_rf(name, rf)
    hm = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, output_dir=prj, project_name='myproject', backend=backend.capitalize(),
        io_type=C.IO_TYPE, part=PARTS[backend], clock_period=C.CLOCK_NS)
    hm.write()
    n_patched = patch_output_loop(prj)
    hm.write = lambda: None
    hm.compile()
    y = C.csim_predict(hm, x, workers=args.workers)
    stock = np.load(os.path.join(C.SCRATCH, f'csim_logits_{name}.npy'))[:len(y)].astype(np.float64)
    params = open(os.path.join(prj, 'firmware', 'parameters.h')).read()
    rec = {'backend': backend, 'config': name, 'rf': rf, 'n': len(y), 'patched_loops': n_patched,
           'n_unit_par': re.findall(r'n_unit_par = (\d+)', params)[:1], 'n_cycles': re.findall(r'n_cycles = (\d+)', params)[:1],
           'max_abs_diff_vs_stock': float(np.abs(y - stock).max()),
           'decisions_equal_stock': float(((y > 0) == (stock > 0)).mean()),
           'stats_vs_fp32_keras': C.paired(kl > 0, y > 0, truth)}
    rows[tag] = rec
    print(tag, json.dumps(rec), flush=True)
    C.dump(rows, os.path.join(HERE, 'results', f'make_projects_n{len(y)}{"_" + args.only if args.only else ""}.json'))
    tb = os.path.join(prj, 'tb_data')
    os.makedirs(tb, exist_ok=True)
    np.savetxt(os.path.join(tb, 'tb_input_features.dat'), x[:100].reshape(100, -1), fmt='%g')
    np.savetxt(os.path.join(tb, 'tb_output_predictions.dat'), y[:100].reshape(100, 1), fmt='%.10g')
    C.dump(cfg, os.path.join(prj, 'hls4ml_config_used.json'))
