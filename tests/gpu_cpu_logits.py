# Created: 2026-09-26
# Last modified: 2026-09-26
"""Logits of one checkpoint on 10,000 validation-block shots, on whatever device is visible.

Run twice, once in a container with the GPU and once without, then compare:

    python tests/gpu_cpu_logits.py --weights W --d 5 --reset-after --unroll --out gpu.npz
    python tests/gpu_cpu_logits.py --weights W --d 5 --reset-after --unroll --out cpu.npz
    python tests/gpu_cpu_logits.py --compare gpu.npz cpu.npz

Uses the repo's model builder, scatter and host-memory input pipeline; validation shots only.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights')
    ap.add_argument('--d', type=int, default=5)
    ap.add_argument('--units', type=int, default=140)
    ap.add_argument('--reset-after', action='store_true')
    ap.add_argument('--unroll', action='store_true')
    ap.add_argument('--n', type=int, default=10_000)
    ap.add_argument('--out')
    ap.add_argument('--compare', nargs=2)
    ap.add_argument('--no-tf32', action='store_true',
                    help='disable TensorFloat-32 matmuls on the GPU (diagnostic)')
    a = ap.parse_args()
    if a.compare:
        g, c = (np.load(f) for f in a.compare)
        diff = float(np.max(np.abs(g['logits'] - c['logits'])))
        sign = float(np.mean(np.sign(g['logits']) == np.sign(c['logits'])))
        print(f"[gpu-cpu] {g['device']} vs {c['device']}: max |logit diff| {diff:.3g}, "
              f"sign agreement {sign:.6f}, n={len(g['logits']):,}  "
              f"{'PASS' if diff < 1e-4 else 'CHECK'} (target ~1e-5)", flush=True)
        return
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    import partitions as P
    from StudentModels import build_gru_student, to_sequence
    from host_data import make_dataset, to_host_uint8
    if a.no_tf32:
        tf.config.experimental.enable_tensor_float_32_execution(False)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=3500)])
    va = P.block('val')
    det = np.load(P.pool_path(a.d, 'formal'))['det_evts'][va[0]:va[0] + a.n]
    x = to_host_uint8(to_sequence(det, a.d, a.d, P.load()['p']))
    m = build_gru_student(d=a.d, rounds=a.d, units=a.units, p=P.load()['p'],
                          reset_after=a.reset_after, unroll=a.unroll)
    m.load_weights(a.weights)
    logits = m.predict(make_dataset(tf, x, None, 10_000), verbose=0).reshape(-1)
    dev = ('GPU' + (' (TF32 off)' if a.no_tf32 else '')) if gpus else 'CPU'
    np.savez(a.out, logits=logits, device=dev)
    print(f"[gpu-cpu] {dev}: {len(logits):,} logits -> {a.out}", flush=True)


if __name__ == '__main__':
    main()
