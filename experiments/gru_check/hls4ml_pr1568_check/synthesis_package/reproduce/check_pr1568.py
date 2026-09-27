# Created: 2026-09-27
# Last modified: 2026-09-27
"""Standalone check: does the installed hls4ml reproduce a Keras GRU model numerically?

Converts a .keras model with whatever hls4ml is installed, compiles the C simulation, runs it
on the given inputs and compares with the expected Keras logits:

    python check_pr1568.py --model ../models/gru_u140_reset_after_False.keras \
        --inputs eval_inputs_10000.npy --expected keras_logits_reset_after_False.npy

Conversion settings (the same as for the recorded results): Vivado backend, io_parallel,
every precision ap_fixed<32,16> (granularity 'name'), GRU activation tables of 65,536 entries
of ap_fixed<32,16> with hls4ml's default input ranges.

Prints the hls4ml version, max and mean |logit_hls - logit_keras|, decision agreement
(decision = logit > 0) and the number of differing decisions, and the reset_after value in
the generated firmware/parameters.h ("not set" if hls4ml writes none).

Needs TensorFlow 2.15 / Keras 2 (the models were saved with tf_keras 2.15) and a C++
compiler for hls4ml's C simulation.
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
