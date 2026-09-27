# Created: 2026-09-27
# Last modified: 2026-09-27
"""Generate the two hls4ml projects for C synthesis: d=5 GRU u140, reset_after=True vs False.

Run in the qdec:tf215-hlsfix image (hls4ml at PR #1568, commit e1721b3):

    python3 convert_for_synthesis.py --out <package_dir>

For each variant it rebuilds the trained model, saves it as a full .keras file, converts it
with hls4ml (Vivado backend, io_parallel, every precision ap_fixed<32,16>, GRU activation
tables 65,536 entries of ap_fixed<32,16> with hls4ml's default input ranges), and writes the
project sources with hls_model.write() only. Nothing is compiled here, so the projects contain
no compiled libraries. A 100-shot testbench (inputs + Keras logits, from the validation block)
is written into each project's tb_data/.

Checkpoints (trained with the same data: 15M FORMAL + 5M TRAINONLY seed 43; seed 0):
  reset_after=False   d5_u140_ntr20M_aug_20260821T154820Z/seed0 (best validation loss)
  reset_after=True    queue_ra1_20M/d5_s0_20260926T174023Z/seed0 (best validation LER;
                      trained with unroll=True, which does not change the equations or weights)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.expanduser('~/FNAL-AI4QC-GRU')
sys.path.insert(0, os.path.join(REPO, 'src'))
PREC = 'ap_fixed<32,16>'
N_TB = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import hls4ml
    import partitions as P
    from StudentModels import build_gru_student, to_sequence

    va = P.block('val')
    x = to_sequence(np.load(P.pool_path(5, 'formal'))['det_evts'][va[0]:va[0] + N_TB], 5, 5,
                    0.004).astype(np.float32)
    R = os.path.expanduser('~/rcnn_threshold/results')
    Q = os.path.join(REPO, 'experiments', 'queue_ra1_20M')
    variants = {
        'gru_u140_reset_after_False': (False, False,
            f'{R}/d5_u140_ntr20M_aug_20260821T154820Z/seed0/ckpt/gru_d5_p004_u140_ext5000000_hard_seed0_ntr15000000_b10000.best.weights.h5'),
        'gru_u140_reset_after_True': (True, True,
            glob.glob(f'{Q}/d5_s0_20260926T174023Z/seed0/ckpt/*.best.weights.h5')[0]),
    }
    os.makedirs(os.path.join(a.out, 'models'), exist_ok=True)
    summary = dict(hls4ml=hls4ml.__version__, tensorflow=tf.__version__, precision=PREC,
                   backend='Vivado', io_type='io_parallel', table_size=65536, variants={})
    for name, (ra, ur, w) in variants.items():
        m = build_gru_student(d=5, rounds=5, units=140, p=0.004, reset_after=ra, unroll=ur)
        m.load_weights(w)
        m.save(os.path.join(a.out, 'models', f'{name}.keras'))
        y = m.predict(x, verbose=0)
        cfg = hls4ml.utils.config_from_keras_model(m, granularity='name', default_precision=PREC,
                                                   backend='Vivado')
        cfg['LayerName']['gru']['TableSize'] = 65536
        cfg['LayerName']['gru']['Precision']['table'] = PREC
        proj = os.path.join(a.out, 'hls_projects', name)
        tbx, tby = os.path.join(a.out, f'_tb_{name}_x.npy'), os.path.join(a.out, f'_tb_{name}_y.npy')
        np.save(tbx, x.reshape(N_TB, -1)); np.save(tby, y)
        hm = hls4ml.converters.convert_from_keras_model(
            m, hls_config=cfg, output_dir=proj, project_name='myproject', backend='Vivado',
            io_type='io_parallel', input_data_tb=tbx, output_data_tb=tby)
        hm.write()
        os.remove(tbx); os.remove(tby)
        gl = [l for l in hm.get_layers() if l.class_name == 'GRU'][0]
        summary['variants'][name] = dict(
            reset_after=ra, keras_params=int(m.count_params()), input_shape=[6, 24],
            apply_reset_gate=gl.get_attr('apply_reset_gate'),
            part=hm.config.get_config_value('Part'), clock_period_ns=hm.config.get_config_value('ClockPeriod'),
            checkpoint=os.path.basename(w))
        print(name, json.dumps(summary['variants'][name]), flush=True)
    with open(os.path.join(a.out, 'generation_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)


if __name__ == '__main__':
    main()
