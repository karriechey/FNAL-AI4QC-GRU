#!/usr/bin/env python3
# from FNAL-QCDecoding-FPGA/export_student_for_hls4ml.py @ d5ee86f
# Created: 2026-08-04
# Last modified: 2026-08-04

"""Export a student as a full Keras .h5 (graph + weights) for hls4ml.

Each export is reloaded and checked to give the same predictions.

python export_student_for_hls4ml.py --out-dir ~/Documents/FNAL-QCDecoding-FPGA/hls4ml_upload
"""
import argparse
import os
import numpy as np


def verify_roundtrip(model, path, x):
    # max |difference| after reload; 0 for a clean FP32 save
    import tensorflow as tf
    reloaded = tf.keras.models.load_model(path, compile=False)
    a = model.predict(x, verbose=0).reshape(-1)
    b = reloaded.predict(x, verbose=0).reshape(-1)
    return float(np.abs(a - b).max()), reloaded


def describe(model):
    # one line per layer: name, type, output shape, params
    lines = []
    for lyr in model.layers:
        try:
            shape = str(lyr.output_shape)
        except AttributeError:
            shape = '?'
        lines.append(f"    {lyr.name:22s} {type(lyr).__name__:18s} {shape:18s} "
                     f"{lyr.count_params():>8,}")
    return '\n'.join(lines)


def export_one(build_kwargs, tag, out_dir, weights=None, reset_after=None):
    import tensorflow as tf
    from StudentModels import build_student, detector_sequence_layout

    student = build_kwargs['student']
    # build_student does not pass p to the GRU builder, so call it directly
    if student == 'gru':
        from StudentModels import build_gru_student
        kw = {k: v for k, v in build_kwargs.items() if k != 'student'}
        model = build_gru_student(**kw)
    else:
        kw = {k: v for k, v in build_kwargs.items() if k != 'p'}
        model = build_student(**kw)

    # reset_after override through the saved config, so Keras builds the layer itself
    if reset_after is not None and student == 'gru':
        cfg = model.get_config()
        for lyr in cfg['layers']:
            if lyr['class_name'] == 'GRU':
                lyr['config']['reset_after'] = reset_after
        model = tf.keras.Model.from_config(cfg)

    if weights:
        if not os.path.exists(weights):
            raise SystemExit(f"[export] MISSING weights {weights}")
        model.load_weights(weights)
        provenance = os.path.basename(weights)
    else:
        # untrained weights: parse test only
        provenance = 'UNTRAINED (structural parse test only)'

    if student == 'gru':
        n_t, n_pos, _ = detector_sequence_layout(build_kwargs.get('d', 5),
                                                 build_kwargs.get('rounds', 3),
                                                 build_kwargs.get('p', 0.010))
        x = np.random.randint(0, 2, (8, n_t, n_pos)).astype(np.float32)
    else:
        n_feat = model.input_shape[-1]
        x = np.random.randint(0, 2, (8, n_feat)).astype(np.float32)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{tag}.h5')
    model.save(path, save_format='h5')

    max_diff, reloaded = verify_roundtrip(model, path, x)
    size_kb = os.path.getsize(path) / 1024

    print(f"  {tag}")
    print(f"    file        {path}  ({size_kb:.1f} KB)")
    print(f"    weights     {provenance}")
    print(f"    input       {model.input_shape}   output {model.output_shape}")
    print(f"    params      {model.count_params():,}")
    print(f"    round-trip  max|pred diff| = {max_diff:.3e}"
          f"{'  OK' if max_diff == 0.0 else '  <-- NONZERO, inspect before uploading'}")
    print(describe(model))
    print()
    return path, max_diff


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir',
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'hls4ml_upload'),
                    help='where to write the uploadable .h5 files')
    ap.add_argument('--d', type=int, default=5)
    ap.add_argument('--p', type=float, default=0.010)
    ap.add_argument('--rounds', type=int, default=3)
    # defaults reproduce the pools_t200k ladder: MLP (128,128) 25,985 params, GRU u64 17,153
    ap.add_argument('--mlp-hidden', type=int, nargs='*', default=[128, 128])
    ap.add_argument('--gru-units', type=int, default=64)
    ap.add_argument('--students', nargs='*', default=['mlp', 'gru'])
    ap.add_argument('--mlp-weights', default=None,
                    help='optional .weights.h5 for the MLP (not needed for a parse test)')
    ap.add_argument('--gru-weights', default=None,
                    help='optional .weights.h5 for the GRU (not needed for a parse test)')
    ap.add_argument('--both-reset-after', action='store_true', default=True,
                    help='export the GRU at reset_after False AND True, isolating it as '
                         'the cause if the GRU fails to parse (default on)')
    ap.add_argument('--single-reset-after', dest='both_reset_after', action='store_false')
    ap.add_argument('--cpu', action='store_true', default=True)
    args = ap.parse_args()

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')
    assert tf.__version__.startswith('2.15'), (
        f'need TF 2.15.x (Keras 2); got TF {tf.__version__}.')

    print(f"[export] full-model .h5 for hls4ml upload -> {args.out_dir}")
    print(f"[export] TF {tf.__version__}\n")

    written = []
    if 'mlp' in args.students:
        written.append(export_one(
            dict(student='mlp', d=args.d, rounds=args.rounds, inputs='evts',
                 hidden=tuple(args.mlp_hidden)),
            'student_mlp_evts_fp32', args.out_dir, weights=args.mlp_weights))

    if 'gru' in args.students:
        base = dict(student='gru', d=args.d, rounds=args.rounds, inputs='evts',
                    units=args.gru_units, hidden=(), p=args.p)
        # reset_after=False: what StudentModels pins, the primary artifact
        written.append(export_one(base, 'student_gru_evts_fp32_reset_after_False',
                                  args.out_dir, weights=args.gru_weights,
                                  reset_after=False))
        if args.both_reset_after:
            # Keras default; diagnostic only, isolates reset_after if only one parses
            written.append(export_one(base, 'student_gru_evts_fp32_reset_after_True',
                                      args.out_dir, weights=None, reset_after=True))

    print("[export] done. Upload these to the hls4ml tab's "
          "'Upload Pre-Trained Model' field:")
    for path, diff in written:
        print(f"    {path}   (round-trip {diff:.0e})")
    print("\n[export] These carry UNTRAINED weights unless --*-weights was passed. That is "
          "correct for a structural parse test -- the graph is what is under test -- but "
          "do NOT read any accuracy or numerical result off them.")


if __name__ == '__main__':
    run()
