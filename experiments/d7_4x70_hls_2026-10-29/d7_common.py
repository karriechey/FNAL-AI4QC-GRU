# Created: 2026-10-29 | Last updated: 2026-10-29
# Shared pieces of the hls4ml gate for the stacked GRU depth_d7_L04_u70 (4 x GRU 70, reset_after=False, Dense 1).
import json
import multiprocessing as mp
import os
import subprocess

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = '/tmp/kchey_qat/d7_4x70'
EVAL_NPZ = os.path.join(SCRATCH, 'eval_d7_15200000_17000000.npz')
RUN = ('/home/kchey/FNAL-AI4QC-GRU-depth/experiments/depth_scaling_d7_2026-10/depth_sweep/'
       'depth_d7_L04_u70_20261021T170721Z')
WEIGHTS = RUN + '/seed0/epochs/epoch078.weights.h5'
PER_SHOT = RUN + '/eval_selected_per_shot.npz'
LOGGED = {'pL': 0.003636, 'mwpm_pL': 0.004291, 'n_params': 113891}
EVAL_START, EVAL_STOP, SEALED_START = 15_200_000, 17_000_000, 17_000_000

D, N_T, N_POS, UNITS = 7, 8, 48, (70, 70, 70, 70)
GRU_NAMES = ['gru', 'gru_1', 'gru_2', 'gru_3']
PART, CLOCK_NS, IO_TYPE, RF = 'xcvu13p-flga2577-2-e', 5.0, 'io_parallel', 70


def build_model(weights=WEIGHTS):
    # same layers and names as StudentModels.build_gru_student(units=(70,)*4), plain tf.keras for hls4ml
    import tensorflow as tf
    inp = tf.keras.Input(shape=(N_T, N_POS), name='syndrome_sequence')
    h = inp
    for i, (u, name) in enumerate(zip(UNITS, GRU_NAMES)):
        h = tf.keras.layers.GRU(u, return_sequences=i < len(UNITS) - 1, reset_after=False, name=name)(h)
    m = tf.keras.Model(inp, tf.keras.layers.Dense(1, name='logit')(h), name='gru_student')
    m.load_weights(weights)
    assert m.count_params() == LOGGED['n_params']
    return m


def load_eval(n=None):
    z = np.load(EVAL_NPZ)
    stop = len(z['truth']) if n is None else n
    idx = z['shot_idx'][:stop]
    assert idx.min() >= EVAL_START and idx.max() < EVAL_STOP <= SEALED_START
    return {k: z[k][:stop] for k in ('x', 'truth', 'keras_logit', 'mwpm_pred', 'shot_idx')}


def paired(ref_pred, new_pred, truth):
    # b = reference (FP32 Keras) right / new wrong, c = new right / reference wrong; exact McNemar
    from scipy.stats import binomtest
    rc, nc = ref_pred == truth, new_pred == truth
    b, c = int((rc & ~nc).sum()), int((~rc & nc).sum())
    return {'n': int(len(truth)), 'errors_keras': int((~rc).sum()), 'errors_hls': int((~nc).sum()),
            'pL_keras': float((~rc).mean()), 'pL_hls': float((~nc).mean()), 'b': b, 'c': c, 'n_disc': b + c,
            'p_exact': float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0,
            'decision_agreement': float((ref_pred == new_pred).mean())}


def int_bits(maxabs):
    # I = floor(log2 max|v|) + 2 (sign included), at least 1
    return max(1, int(np.floor(np.log2(maxabs))) + 2) if maxabs > 0 else 1


def base_config(model, precision='ap_fixed<32,16>'):
    import hls4ml
    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name', default_precision=precision, backend='Vivado')
    cfg['Model']['Strategy'] = 'Resource'
    cfg['Model']['ReuseFactor'] = RF
    for name in GRU_NAMES + ['logit']:
        cfg['LayerName'][name]['Strategy'] = 'Resource'
        cfg['LayerName'][name]['ReuseFactor'] = RF
    for name in GRU_NAMES:
        cfg['LayerName'][name]['static'] = True
    return cfg


def convert(model, cfg, prj):
    import hls4ml
    hm = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, output_dir=prj, project_name='myproject', backend='Vivado',
        io_type=IO_TYPE, part=PART, clock_period=CLOCK_NS)
    hm.write()
    return hm


def hls4ml_record():
    import hls4ml
    root = os.path.dirname(os.path.dirname(hls4ml.__file__))
    git = lambda *a: subprocess.run(['git', '-C', root, *a], capture_output=True, text=True).stdout.strip()
    return {'hls4ml_version': hls4ml.__version__, 'hls4ml_path': root, 'hls4ml_commit': git('rev-parse', 'HEAD'),
            'contains_5376da23e': subprocess.run(['git', '-C', root, 'merge-base', '--is-ancestor', '5376da23e', 'HEAD']).returncode == 0,
            'hls4ml_local_changes': git('status', '--porcelain', '--untracked-files=no') or 'none'}


_HM = {}


def _work(task):
    a, xb = task
    return a, np.asarray(_HM['m'].predict(np.ascontiguousarray(xb)), float).reshape(-1)


def csim_predict(hm, x, workers=48, chunk=None):
    # chunk sized so every worker gets work on small shot counts too
    _HM['m'] = hm
    x = np.ascontiguousarray(x.astype(np.float32))
    chunk = chunk or max(50, min(2000, -(-len(x) // workers)))
    tasks = [(a, x[a:a + chunk]) for a in range(0, len(x), chunk)]
    y = np.empty(len(x))
    with mp.get_context('fork').Pool(min(workers, len(tasks))) as pool:
        for a, yy in pool.imap_unordered(_work, tasks):
            y[a:a + len(yy)] = yy
    return y


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=str)
