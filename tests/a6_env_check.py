# Created: 2026-09-26
# Last modified: 2026-09-26

"""A6: environment check for one grace1 environment.

Reports CPU, GPU, CUDA and package versions, decodes a d=3 circuit with stim + pymatching, and
checks TensorFlow sees the GPU. Writes experiments/a6_env/<UTC stamp>_<env>.json.

~/tfenv/bin/python tests/a6_env_check.py --env tfenv
podman run ... qdec:tf215 python3 tests/a6_env_check.py --env qdec-tf215
"""
import argparse
import datetime
import importlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

PACKAGES = [('numpy', 'numpy'), ('scipy', 'scipy'), ('stim', 'stim'),
            ('pymatching', 'pymatching'), ('sinter', 'sinter'),
            ('tensorflow', 'tensorflow'), ('qkeras', 'qkeras'), ('hls4ml', 'hls4ml'),
            ('pyyaml', 'yaml'), ('pytest', 'pytest')]


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT,
                                       timeout=60).decode().strip()
    except Exception as exc:
        return f'unavailable ({type(exc).__name__})'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True, help='label for this environment')
    a = ap.parse_args()
    rep = dict(env=a.env, host=platform.node(), machine=platform.machine(),
               cpu_count=os.cpu_count(), python=platform.python_version(),
               cpu_model=sh("lscpu | grep -m1 'Model name' | sed 's/.*: *//'"),
               gpu=sh('nvidia-smi --query-gpu=name,memory.total,driver_version '
                      '--format=csv,noheader'),
               cuda_driver=sh("nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"),
               nvcc=sh('nvcc --version | tail -1'), packages={})
    for dist, mod in PACKAGES:
        try:
            ver = md.version(dist)
        except md.PackageNotFoundError:
            rep['packages'][dist] = 'NOT INSTALLED'
            continue
        try:
            importlib.import_module(mod)
            rep['packages'][dist] = ver
        except Exception as exc:
            rep['packages'][dist] = f'{ver} IMPORT FAILED: {type(exc).__name__}: {exc}'

    # stim + pymatching on a d=3 circuit
    try:
        import numpy as np
        import pymatching
        from circuit_generators import build_circuit
        c = build_circuit(3, 0.004, 3)
        det, obs = c.compile_detector_sampler(seed=1).sample(100_000, separate_observables=True)
        m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
        pred = m.decode_batch(det)
        ler = float((pred[:, 0] != obs[:, 0]).mean())
        rep['d3_decode'] = dict(ok=0 < ler < 0.05, shots=100_000, mwpm_ler=ler,
                                detectors=c.num_detectors)
    except Exception as exc:
        rep['d3_decode'] = dict(ok=False, error=f'{type(exc).__name__}: {exc}')

    # TensorFlow devices and whether CUDA was compiled in.
    try:
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
        import tensorflow as tf
        rep['tf'] = dict(gpus=[g.name for g in tf.config.list_physical_devices('GPU')],
                         built_with_cuda=tf.test.is_built_with_cuda(),
                         build_info={k: str(v) for k, v in tf.sysconfig.get_build_info().items()
                                     if k in ('cuda_version', 'cudnn_version')})
    except Exception as exc:
        rep['tf'] = dict(error=f'{type(exc).__name__}: {exc}')

    stamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out = os.path.join(ROOT, 'experiments', 'a6_env', f'{stamp}_{a.env}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as fh:
        json.dump(rep, fh, indent=1)
    for k, v in rep.items():
        if k == 'packages':
            for p, pv in v.items():
                print(f'[A6] {a.env}  {p:12s} {pv}')
        else:
            print(f'[A6] {a.env}  {k}: {v}')
    print(f'[A6] wrote {out}')


if __name__ == '__main__':
    main()
