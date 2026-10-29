# Created: 2026-10-29 | Last updated: 2026-10-29
# Uncorrelated vs correlated MWPM (PyMatching enable_correlations) vs the best GRU, on the shots of an
# existing eval-block per-shot dump (detection events, truth and GRU decisions are read from it; nothing
# is regenerated). Each decoder runs in a forked child, so its CPU time and peak RSS are its own.
# CPU only.
#   ~/tfenv/bin/python compare_mwpm.py --d 5 --dump <teacher_eval_per_shot.npz> --gru f5w_p004 [--n 10000]
import argparse
import hashlib
import json
import os
import pickle
import platform
import resource
import sys
import tempfile
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'src'))
from circuit_generators import build_circuit

ap = argparse.ArgumentParser()
ap.add_argument('--d', type=int, required=True)
ap.add_argument('--p', type=float, default=0.004)
ap.add_argument('--dump', required=True)
ap.add_argument('--gru', required=True, help='name of the GRU whose decisions are in the dump')
ap.add_argument('--n', type=int, default=None, help='first n shots only (pilot)')
ap.add_argument('--out', default=os.path.join(HERE, 'results'))
args = ap.parse_args()

z = np.load(args.dump)
n = len(z['truth']) if args.n is None else args.n
det = np.ascontiguousarray(z['det_evts'][:n].astype(np.uint8))
truth = z['truth'][:n].astype(bool)
gru_pred = z['student_pred'][:n].astype(bool)
stored_mwpm = z['mwpm_pred'][:n].astype(bool)
shot_idx = z['shot_idx'][:n]
assert shot_idx.min() >= 15_200_000 and shot_idx.max() < 17_000_000   # eval block, never the sealed block

circ = build_circuit(args.d, args.p, args.d)
assert circ.num_detectors == det.shape[1], (circ.num_detectors, det.shape)
dem = circ.detector_error_model(decompose_errors=True)
dem_sha = hashlib.sha256(str(dem).encode()).hexdigest()


def run_child(fn):
    # fork; the child runs fn and pickles its result; returns (result, wall s, cpu s, peak RSS MiB of the child)
    fd, path = tempfile.mkstemp(suffix='.pkl')
    os.close(fd)
    t0 = time.time()
    pid = os.fork()
    if pid == 0:
        try:
            with open(path, 'wb') as f:
                pickle.dump(fn(), f)
            os._exit(0)
        except BaseException as e:
            with open(path, 'wb') as f:
                pickle.dump({'error': repr(e)}, f)
            os._exit(1)
    _, status, ru = os.wait4(pid, 0)
    wall = time.time() - t0
    res = pickle.load(open(path, 'rb'))
    os.remove(path)
    if status != 0:
        raise SystemExit(f'child failed: {res}')
    return res, wall, ru.ru_utime + ru.ru_stime, ru.ru_maxrss / 1024


def decoder(correlated):
    def fn():
        import pymatching
        kw = {'enable_correlations': True} if correlated else {}
        t0 = time.process_time()
        m = pymatching.Matching.from_detector_error_model(dem, **kw)
        t1 = time.process_time()
        pred = m.decode_batch(det, bit_packed_shots=False, bit_packed_predictions=False, **kw)
        t2 = time.process_time()
        return {'pred': np.asarray(pred).reshape(-1).astype(bool), 'build_cpu_s': t1 - t0, 'decode_cpu_s': t2 - t1}
    return fn


import pymatching
import stim
_, _, _, baseline_mib = run_child(lambda: {'noop': True})   # RSS of a child that only holds the loaded arrays
runs = {}
for name, corr in (('mwpm', False), ('mwpm_correlated', True)):
    res, wall, cpu, rss = run_child(decoder(corr))
    runs[name] = {'pred': res['pred'], 'wall_s': wall, 'cpu_s': cpu, 'peak_rss_mib': rss,
                  'build_cpu_s': res['build_cpu_s'], 'decode_cpu_s': res['decode_cpu_s'],
                  'decode_us_per_shot': 1e6 * res['decode_cpu_s'] / n}
    print(f"[{name}] wall {wall:.1f} s  cpu {cpu:.1f} s  build {res['build_cpu_s']:.2f} s  "
          f"decode {res['decode_cpu_s']:.1f} s ({runs[name]['decode_us_per_shot']:.1f} us/shot)  "
          f"peak RSS {rss:.0f} MiB (no-op child {baseline_mib:.0f} MiB)", flush=True)


def wilson(k, n, zc=1.959964):
    p = k / n
    den = 1 + zc ** 2 / n
    c = (p + zc ** 2 / (2 * n)) / den
    h = zc * np.sqrt(p * (1 - p) / n + zc ** 2 / (4 * n * n)) / den
    return [c - h, c + h]


def mcnemar(a_pred, b_pred):
    # b = A right / B wrong, c = B right / A wrong; exact two-sided binomial
    from scipy.stats import binomtest
    ar, br = a_pred == truth, b_pred == truth
    b, c = int((ar & ~br).sum()), int((~ar & br).sum())
    return {'b_A_right_B_wrong': b, 'c_B_right_A_wrong': c, 'n_disc': b + c,
            'p_exact': float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0}


preds = {'mwpm': runs['mwpm']['pred'], 'mwpm_correlated': runs['mwpm_correlated']['pred'], 'gru': gru_pred}
err = {k: int((v != truth).sum()) for k, v in preds.items()}
pL = {k: e / n for k, e in err.items()}
out = {
    'd': args.d, 'p': args.p, 'rounds': args.d, 'n_shots': n, 'shot_range': [int(shot_idx.min()), int(shot_idx.max()) + 1],
    'gru': args.gru, 'dump': os.path.abspath(args.dump), 'dem_sha256': dem_sha,
    'uncorrelated_equals_stored_mwpm': bool(np.array_equal(preds['mwpm'], stored_mwpm)),
    'errors': err, 'pL': pL, 'pL_wilson95': {k: wilson(e, n) for k, e in err.items()},
    'x_mwpm': {'gru_over_mwpm': pL['gru'] / pL['mwpm'], 'gru_over_mwpm_correlated': pL['gru'] / pL['mwpm_correlated'],
               'mwpm_correlated_over_mwpm': pL['mwpm_correlated'] / pL['mwpm']},
    'relative_improvement_pct': {'gru_vs_mwpm': 100 * (1 - pL['gru'] / pL['mwpm']),
                                 'gru_vs_mwpm_correlated': 100 * (1 - pL['gru'] / pL['mwpm_correlated']),
                                 'mwpm_correlated_vs_mwpm': 100 * (1 - pL['mwpm_correlated'] / pL['mwpm'])},
    'mcnemar': {'gru_vs_mwpm_correlated': mcnemar(preds['gru'], preds['mwpm_correlated']),
                'gru_vs_mwpm': mcnemar(preds['gru'], preds['mwpm']),
                'mwpm_correlated_vs_mwpm': mcnemar(preds['mwpm_correlated'], preds['mwpm'])},
    'cost': {k: {kk: vv for kk, vv in v.items() if kk != 'pred'} for k, v in runs.items()},
    'noop_child_peak_rss_mib': baseline_mib,
    'environment': {'pymatching': pymatching.__version__, 'stim': stim.__version__, 'numpy': np.__version__,
                    'python': platform.python_version(), 'host': platform.node(), 'machine': platform.machine(),
                    'cpu_threads_used': 1},
}
os.makedirs(args.out, exist_ok=True)
tag = f"d{args.d}_n{n}"
with open(os.path.join(args.out, f'compare_{tag}.json'), 'w') as f:
    json.dump(out, f, indent=1)
np.savez_compressed(os.path.join(args.out, f'per_shot_{tag}.npz'), shot_idx=shot_idx,
                    mwpm_pred=preds['mwpm'], mwpm_correlated_pred=preds['mwpm_correlated'], gru_pred=gru_pred, truth=truth)
print(json.dumps({k: out[k] for k in ('d', 'n_shots', 'uncorrelated_equals_stored_mwpm', 'errors', 'x_mwpm',
                                      'relative_improvement_pct', 'mcnemar')}, indent=1), flush=True)
