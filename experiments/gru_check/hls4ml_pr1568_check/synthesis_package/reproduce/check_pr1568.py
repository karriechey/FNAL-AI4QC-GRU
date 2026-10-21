# Created: 2026-09-27
# Last modified: 2026-09-27

"""Does the installed hls4ml reproduce a Keras GRU model numerically?

Converts with Vivado, io_parallel, ap_fixed<32,16> and 65,536-entry GRU tables, runs C-sim, and
prints max and mean |logit_hls - logit_keras|, decision agreement (logit > 0), and the
reset_after value in firmware/parameters.h. Needs TF 2.15 / Keras 2 and a C++ compiler.

python check_pr1568.py --model ../models/gru_u140_reset_after_False.keras \
    --inputs eval_inputs_10000.npy --expected keras_logits_reset_after_False.npy
"""
import argparse
import os
import re
import tempfile

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='.keras model')
    ap.add_argument('--inputs', required=True, help='.npy, shape (N, 6, 24)')
    ap.add_argument('--expected', required=True, help='.npy of Keras logits, shape (N,)')
    ap.add_argument('--precision', default='ap_fixed<32,16>')
    ap.add_argument('--table-size', type=int, default=65536)
    ap.add_argument('--out', default=None, help='hls4ml project folder (default: a temp dir)')
    a = ap.parse_args()

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf
    import hls4ml

    x = np.ascontiguousarray(np.load(a.inputs), dtype=np.float32)
    y_expected = np.load(a.expected).reshape(-1)
    model = tf.keras.models.load_model(a.model, compile=False)
    y_keras = model.predict(x, batch_size=10000, verbose=0).reshape(-1)
    gru = [l for l in model.layers if l.__class__.__name__ == 'GRU'][0]

    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name',
                                               default_precision=a.precision, backend='Vivado')
    cfg['LayerName'][gru.name]['TableSize'] = a.table_size
    cfg['LayerName'][gru.name]['Precision']['table'] = a.precision
    out = a.out or tempfile.mkdtemp(prefix='hls_check_')
    hm = hls4ml.converters.convert_from_keras_model(model, hls_config=cfg, output_dir=out,
                                                    backend='Vivado', io_type='io_parallel')
    hm.compile()
    y_hls = np.asarray(hm.predict(x)).reshape(-1)
    params = open(os.path.join(out, 'firmware', 'parameters.h')).read()
    m = re.search(r'static const bool reset_after = (\w+);', params)

    d = np.abs(y_hls - y_expected)
    differ = int(np.sum((y_hls > 0) != (y_expected > 0)))
    print(f'hls4ml                     {hls4ml.__version__}')
    print(f'model                      {os.path.basename(a.model)} '
          f'(reset_after={gru.get_config()["reset_after"]}, units={gru.units})')
    print(f'shots                      {len(x):,}')
    print(f'Keras here vs expected     max |d| {np.max(np.abs(y_keras - y_expected)):.3g}')
    print(f'max  |logit_hls - keras|   {d.max():.3f}')
    print(f'mean |logit_hls - keras|   {d.mean():.4f}')
    print(f'decision agreement         {100 * (1 - differ / len(x)):.2f} %  '
          f'({differ:,} / {len(x):,} decisions differ)')
    print(f'generated reset_after      {m.group(1) if m else "not set"}')
    print(f'hls4ml project             {out}')


if __name__ == '__main__':
    main()
