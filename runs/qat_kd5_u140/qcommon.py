# Created: 2026-10-28 | Last updated: 2026-10-28
# Shared pieces of the kd5_u140 quantization study: paths, the FP32 parent, eval-block data,
# paired statistics, the precision groups and the hls4ml configuration built from them.
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, 'src'))

SCRATCH = os.environ.get('QAT_SCRATCH', '/tmp/kchey_qat')
DATA = os.path.join(SCRATCH, 'data')
EVAL_NPZ = os.path.join(DATA, 'kd5_u140_eval_block_15200000_17000000.npz')

D, ROUNDS, P = 5, 5, 0.004
UNITS, N_T, N_POS = 140, 6, 24
N_PARAMS = 69441
POOL = '/home/kchey/pools_d5_p004_19M/data_d5_p0.004_r5_FORMAL.npz'
EVAL_START, EVAL_STOP = 15_200_000, 17_000_000
SEALED_START = 17_000_000

PARENT_RUN = '/home/kchey/FNAL-AI4QC-GRU-kd/experiments/kd_pilot_p004_2026-10/runs/kd5_u140_20261018T044716Z'
PARENT_WEIGHTS = PARENT_RUN + '/seed0/epochs/epoch079.weights.h5'
PARENT_PER_SHOT = PARENT_RUN + '/eval_selected_per_shot.npz'
# recorded in eval_selected.log and GRU_distillation_runs_2026-10-19.csv
RECORDED = {'p_L': 0.005148, 'mwpm_p_L': 0.007571, 'ratio': 0.6800}

TEACHER_WEIGHTS = '/home/kchey/FNAL-AI4QC-GRU-kd/experiments/kd_pilot_p004_2026-10/teachers/f5w_p004_epoch80.weights.h5'
TEACHER_UNITS, TEACHER_PARAMS = 280, 256481
FRESH_SEED_BASE = 3 * 10 ** 12

# hls4ml synthesis target (the GRU70 runs: experiments/u70_hls_main_2ed80fb9_2026-10-03)
PART, CLOCK_NS, IO_TYPE = 'xcvu13p-flga2577-2-e', 5.0, 'io_parallel'
TABLE_SIZE = 1024
ACCUM_BITS = 24   # accumulators are held wide on both sweep axes (SUMMARY.md, decisions)

# Precision groups. Each maps onto hls4ml config keys (layer, precision key).
WEIGHT_GROUPS = ['gru_weight', 'gru_recurrent_weight', 'gru_bias', 'logit_weight', 'logit_bias']
STATE_GROUPS = ['gru_state', 'gru_table']
ACCUM_GROUPS = ['gru_accum', 'logit_accum']
HLS_KEYS = {
    'gru_weight': [('gru', 'weight')],
    'gru_recurrent_weight': [('gru', 'recurrent_weight')],
    'gru_bias': [('gru', 'bias'), ('gru', 'recurrent_bias')],
    'gru_state': [('gru', 'result')],
    'gru_table': [('gru', 'table_t')],   # layer attribute, not a Precision key
    'gru_accum': [('gru', 'accum')],
    'logit_weight': [('logit', 'weight')],
    'logit_bias': [('logit', 'bias')],
    'logit_accum': [('logit', 'accum'), ('logit', 'result'), ('logit_linear', 'result')],
}


def int_bits_from_max(maxabs, headroom=1):
    # signed integer bits (sign included) that hold maxabs, plus headroom
    base = max(1, math.floor(math.log2(maxabs)) + 2) if maxabs > 0 else 1
    return base + headroom


def spec_for(int_bits, weight_bits, state_bits, accum_bits=ACCUM_BITS, overrides=None):
    # {group: (W, I)}; W >= 64 means float for that group
    spec = {}
    for g in WEIGHT_GROUPS:
        spec[g] = (weight_bits, int_bits[g])
    for g in STATE_GROUPS:
        spec[g] = (state_bits, int_bits[g])
    for g in ACCUM_GROUPS:
        spec[g] = (accum_bits, int_bits[g])
    spec.update(overrides or {})
    return spec


def ap_type(w, i, mode='RND,SAT'):
    return f'ap_fixed<{w},{i},{mode.replace("RND", "AP_RND").replace("SAT", "AP_SAT")}>' if mode else f'ap_fixed<{w},{i}>'


def hls_config(model, spec, strategy='Resource', reuse_factor=1, table_size=TABLE_SIZE, mode='RND,SAT',
               input_type='ap_uint<1>', trace=False):
    import hls4ml
    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name', default_precision='ap_fixed<32,16>',
                                               backend='Vivado')
    cfg['Model']['Strategy'] = strategy
    cfg['Model']['ReuseFactor'] = reuse_factor
    for name in ('gru', 'logit'):
        cfg['LayerName'][name]['Strategy'] = strategy
        cfg['LayerName'][name]['ReuseFactor'] = reuse_factor
    cfg['LayerName']['gru']['static'] = True
    cfg['LayerName']['gru']['table_size'] = table_size
    cfg['LayerName']['syndrome_sequence']['Precision']['result'] = input_type
    for g, (w, i) in spec.items():
        for layer, key in HLS_KEYS[g]:
            if key == 'table_t':
                cfg['LayerName'][layer]['table_t'] = ap_type(w, i, mode)
            else:
                cfg['LayerName'][layer].setdefault('Precision', {})[key] = ap_type(w, i, mode)
    if trace:
        for n in ('gru', 'logit'):
            cfg['LayerName'][n]['Trace'] = True
    return cfg


def convert(model, cfg, output_dir):
    import hls4ml
    hm = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, output_dir=output_dir, project_name='myproject', backend='Vivado',
        io_type=IO_TYPE, part=PART, clock_period=CLOCK_NS)
    hm.write()
    return hm


def hls4ml_record():
    import subprocess
    import hls4ml
    root = os.path.dirname(os.path.dirname(hls4ml.__file__))
    commit = subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', root, 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True).stdout.strip()
    return {'hls4ml_version': hls4ml.__version__, 'hls4ml_path': root, 'hls4ml_commit': commit,
            'hls4ml_local_changes': dirty or 'none'}


def build_fp32(weights=PARENT_WEIGHTS):
    # same layers as StudentModels.build_gru_student (names gru, logit), without the stim import
    import tensorflow as tf
    inp = tf.keras.Input(shape=(N_T, N_POS), name='syndrome_sequence')
    h = tf.keras.layers.GRU(UNITS, reset_after=False, name='gru')(inp)
    m = tf.keras.Model(inp, tf.keras.layers.Dense(1, name='logit')(h), name='gru_student')
    m.load_weights(weights)
    assert m.count_params() == N_PARAMS and m.get_layer('gru').reset_after is False
    return m


def load_eval(n=None):
    z = np.load(EVAL_NPZ)
    stop = len(z['truth']) if n is None else n
    idx = z['shot_idx'][:stop]
    assert idx.min() >= EVAL_START and idx.max() < EVAL_STOP <= SEALED_START
    return {k: z[k][:stop] for k in ('x', 'truth', 'mwpm_pred', 'fp32_logit', 'shot_idx')}


def paired(ref_pred, new_pred, truth):
    # b = reference right / new wrong, c = new right / reference wrong; exact two-sided McNemar
    from scipy.stats import binomtest
    rc, nc = ref_pred == truth, new_pred == truth
    b, c = int((rc & ~nc).sum()), int((~rc & nc).sum())
    return {'n': int(len(truth)), 'errors_ref': int((~rc).sum()), 'errors_new': int((~nc).sum()),
            'b': b, 'c': c, 'n_disc': b + c,
            'p_exact': float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0,
            'agreement': float((ref_pred == new_pred).mean())}


def score(pred, d):
    # p_L, ratio to FP32 and MWPM on the same shots, McNemar vs the FP32 parent
    truth, fp = d['truth'].astype(bool), d['fp32_logit'] > 0
    mw = d['mwpm_pred'].astype(bool)
    st = paired(fp, pred, truth)
    ler, ler_fp, ler_mw = float((pred != truth).mean()), float((fp != truth).mean()), float((mw != truth).mean())
    return {'ler': ler, 'ler_fp32': ler_fp, 'ler_mwpm': ler_mw, 'ler_over_fp32': ler / ler_fp,
            'ler_over_mwpm': ler / ler_mw, 'mcnemar_vs_fp32': st}


def acceptable(row):
    # pre-registered rule (NOTES.md): p_L <= 1.05 x FP32 and p_L < MWPM, same eval-block shots
    return row['ler'] <= 1.05 * row['ler_fp32'] and row['ler'] < row['ler_mwpm']


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=str)
