# Created: 2026-10-03 | Last updated: 2026-10-03
"""Shared pieces for the u70 GRU hls4ml baseline on hls4ml main 2ed80fb9.

Model: d=5 r=5 p=0.004 GRU, 70 units, reset_after=False, 20,021 parameters
(run gkd20m_u70_hard_seed0, trained on the EAF x86 FORMAL pool, gen_seed 42).

Data: the FORMAL evaluation block [15,200,000, 17,000,000) of that same EAF pool
(1,800,000 shots). The sealed block [17,000,000, 19,000,000) is not in the data file,
so nothing here can read it.

Every script imports this module so that the model rebuild, the data loading, the hls4ml
version record and the paired statistics are defined once.
"""
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import json
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
MODEL_FILE = os.path.join(DATA, 'gru_u70_reset_after_False.keras')
EVAL_FILE = os.path.join(DATA, 'u70_eval_block_15200000_17000000.npz')
EVAL_START, EVAL_STOP = 15_200_000, 17_000_000
SEALED_START = 17_000_000
REQUIRED_HLS4ML_COMMIT = '2ed80fb9'

# Synthesis target shared by every configuration in this study.
PART, CLOCK_NS, IO_TYPE = 'xcvu13p-flga2577-2-e', 5.0, 'io_parallel'
RESULT_LABEL = 'upper-bound baseline, <32,16> PTQ'


def hls4ml_version_record():
    """Return the installed hls4ml version and the git commit it was built from.

    pip/uv record the commit of a git install in direct_url.json; reading it proves which
    source the import comes from. Stops the script if the commit is not 2ed80fb9."""
    import hls4ml
    from importlib.metadata import distribution
    dist = distribution('hls4ml')
    direct = dist.read_text('direct_url.json')
    info = json.loads(direct) if direct else {}
    commit = info.get('vcs_info', {}).get('commit_id', '')
    rec = {'hls4ml_version': hls4ml.__version__, 'hls4ml_file': hls4ml.__file__,
           'hls4ml_install_url': info.get('url', ''), 'hls4ml_commit': commit}
    if not commit.startswith(REQUIRED_HLS4ML_COMMIT):
        sys.exit(f'hls4ml is not installed from main at {REQUIRED_HLS4ML_COMMIT}: {rec}')
    return rec


def patch_tag():
    """'' for unmodified hls4ml main; '_kernelfix' when the recurrent kernel-choice fix is applied
    (patch_recurrent_kernel_choice.py), so patched results can never share a file name."""
    import hls4ml
    f = os.path.join(os.path.dirname(hls4ml.__file__), 'backends', 'vivado', 'passes', 'recurrent_templates.py')
    return '_kernelfix' if 'LOCAL FIX 2026-10-03' in open(f).read() else ''


def generated_reuse_factors(prj):
    """Reuse factor and dense kernel of every multiply, read from the generated parameters.h."""
    import re
    params_h = open(os.path.join(prj, 'firmware', 'parameters.h')).read()
    out = {}
    for name, body in re.findall(r'struct (config\w+) : nnet::dense_config \{(.*?)\n\};', params_h, re.S):
        g = lambda k: (re.findall(rf'static const \w+ {k} = ([^;]+);', body) or [None])[0]
        out[name] = {'n_in': g('n_in'), 'n_out': g('n_out'), 'reuse_factor': g('reuse_factor'),
                     'kernel': (re.findall(r'using kernel = ([\w:]+)<', body) or [None])[0]}
    return out


def environment_record():
    """Versions of everything that affects the numbers, for every output file."""
    import platform
    import tensorflow as tf
    import tf_keras
    rec = hls4ml_version_record()
    rec['local_patch'] = patch_tag().lstrip('_') or 'none'
    gxx = subprocess.run(['g++', '--version'], capture_output=True, text=True).stdout.splitlines()
    rec.update({'host': platform.node(), 'machine': platform.machine(), 'python': sys.version.split()[0],
                'tensorflow': tf.__version__, 'tf_keras': tf_keras.__version__, 'numpy': np.__version__,
                'gxx': gxx[0] if gxx else ''})
    return rec


def build_model():
    """Load the tf_keras-saved file and copy its weights into the same architecture built with
    tf.keras, which is what hls4ml parses.

    tf.keras.models.load_model in TF 2.15 loads this tf_keras file without an error but with
    wrong predictions, so the file is always opened with tf_keras."""
    import tensorflow as tf
    import tf_keras
    src = tf_keras.models.load_model(MODEL_FILE, compile=False)
    gru = [l for l in src.layers if isinstance(l, tf_keras.layers.GRU)][0]
    dense = [l for l in src.layers if isinstance(l, tf_keras.layers.Dense)][0]
    inp = tf.keras.Input(shape=(6, 24), name='syndrome_sequence')
    h = tf.keras.layers.GRU(70, activation='tanh', recurrent_activation='sigmoid', reset_after=False,
                            return_sequences=False, name='gru')(inp)
    model = tf.keras.Model(inp, tf.keras.layers.Dense(1, name='logit')(h))
    model.get_layer('gru').set_weights(gru.get_weights())
    model.get_layer('logit').set_weights(dense.get_weights())
    assert model.count_params() == 20021 and model.get_layer('gru').reset_after is False
    return model


def load_eval(n=None, start=0):
    """Shots [start, start+n) of the evaluation block, as float32 GRU inputs (N, 6, 24),
    truth (bool), MWPM prediction (bool) and the Keras logits stored by the EAF run."""
    z = np.load(EVAL_FILE)
    stop = len(z['truth']) if n is None else start + n
    idx = z['shot_idx'][start:stop]
    # Guard: every shot must lie inside the evaluation block, never in the sealed block.
    assert idx.min() >= EVAL_START and idx.max() < EVAL_STOP < SEALED_START + 1
    return {'x': z['x'][start:stop].astype(np.float32), 'truth': z['truth'][start:stop].astype(bool),
            'mwpm_pred': z['mwpm_pred'][start:stop].astype(bool),
            'keras_logit_eaf': z['keras_logit_eaf'][start:stop], 'shot_idx': idx,
            'source_flips_sha256': str(z['source_flips_sha256'])}


def paired_stats(ref_pred, new_pred, truth):
    """Logical error rates of two decoders on identical shots plus the exact McNemar test.

    b = shots the reference gets right and the new decoder gets wrong,
    c = shots the new decoder gets right and the reference gets wrong.
    p_exact is the two-sided binomial test of b against n_discordant / 2."""
    from scipy.stats import binomtest
    rc, nc = ref_pred == truth, new_pred == truth
    b, c = int((rc & ~nc).sum()), int((~rc & nc).sum())
    n = len(truth)
    return {'n': n, 'errors_ref': int((~rc).sum()), 'errors_new': int((~nc).sum()),
            'p_L_ref': float((~rc).mean()), 'p_L_new': float((~nc).mean()),
            'b_ref_only_right': b, 'c_new_only_right': c, 'n_discordant': b + c,
            'p_exact': float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0,
            'decision_match_rate': float((ref_pred == new_pred).mean())}


def base_config(model, precision='ap_fixed<32,16>', strategy='Resource', reuse_factor=70):
    """hls4ml config shared by every configuration: Vivado backend, per-layer granularity,
    static GRU, one default precision for every type that the profiling does not set."""
    import hls4ml
    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name', default_precision=precision,
                                               backend='Vivado')
    cfg['Model']['Strategy'] = strategy
    cfg['Model']['ReuseFactor'] = reuse_factor
    for name in ('gru', 'logit'):
        cfg['LayerName'][name]['Strategy'] = strategy
        cfg['LayerName'][name]['ReuseFactor'] = reuse_factor
    cfg['LayerName']['gru']['static'] = True
    return cfg


def convert(model, cfg, output_dir):
    import hls4ml
    hm = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, output_dir=output_dir, project_name='myproject', backend='Vivado',
        io_type=IO_TYPE, part=PART, clock_period=CLOCK_NS)
    hm.write()
    return hm


def csim_predict(hm, x, chunk=20000):
    """hls_model.predict in chunks (keeps the ctypes buffers small)."""
    out = [np.asarray(hm.predict(np.ascontiguousarray(x[i:i + chunk])), float).reshape(-1)
           for i in range(0, len(x), chunk)]
    return np.concatenate(out)
