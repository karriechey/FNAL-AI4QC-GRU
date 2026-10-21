# Created: 2026-09-26
# Last modified: 2026-09-26

"""Checks for several --extra-train-pool files in train_student.py. CPU only.

order  FORMAL [0, 20k) + seed 43 [0, 5k) + seed 44 [0, 3k): the training arrays match the
       expected concatenation row by row
dupe   the same seed twice is refused (same file twice, or a link to it)

python tests/multi_pool_checks.py order --out <dir>
python tests/multi_pool_checks.py dupe  --out <dir>
"""
import json
import os
import runpy
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, SRC)
D = 7
POOLS = os.path.expanduser('~/pools_d7_p004_19M')
FORMAL = f'{POOLS}/data_d7_p0.004_r7_FORMAL.npz'
S43 = f'{POOLS}/data_d7_p0.004_r7_TRAINONLY_seed43.npz'
S44 = f'{POOLS}/data_d7_p0.004_r7_TRAINONLY_seed44.npz'


def base_args(out, tag, ntr):
    return ['--student', 'gru', '--inputs', 'evts', '--hidden', '--units', '140',
            '--reset-after', '--unroll', '--d', '7', '--p', '0.004', '--rounds', '7',
            '--pool', FORMAL, '--n-train', str(ntr), '--val-start', '15000000',
            '--val-n', '20000', '--eval-start', '15200000', '--n-test', '20000',
            '--alpha', '1.0', '--temperature', '1.0', '--batch-size', '10000', '--lr', '0.003',
            '--epochs', '1', '--no-early-stopping', '--seed', '0', '--cpu',
            '--ckpt-dir', f'{out}/{tag}/ckpt', '--out-dir', f'{out}/{tag}/runs', '--tag', tag]


def run_trainer(args):
    sys.argv = [os.path.join(SRC, 'train_student.py')] + args
    os.chdir(SRC)
    try:
        runpy.run_path(os.path.join(SRC, 'train_student.py'), run_name='__main__')
        return None
    except SystemExit as e:
        return e.code
    except Exception as e:
        return f'{type(e).__name__}: {e}'


class Captured(Exception):
    pass


def order(out):
    import host_data
    got = {}
    real = host_data.make_dataset

    def capture(tf, x_u8, y=None, batch_size=10_000, shuffle_seed=None, to_gpu=True):
        if shuffle_seed is not None:           # the training pipeline
            got['x'], got['y'] = np.array(x_u8), np.array(y)
            raise Captured()
        return real(tf, x_u8, y, batch_size, shuffle_seed, to_gpu)
    host_data.make_dataset = capture
    counts = (20_000, 5_000, 3_000)
    res = run_trainer(base_args(out, 'order', counts[0]) +
                      ['--extra-train-pool', S43, S44, '--extra-train-n', str(counts[1]),
                       str(counts[2])])
    if 'x' not in got:
        print(f'[multi] FAIL order: training arrays not captured (trainer returned {res})')
        return False
    from StudentModels import to_sequence
    parts = [(FORMAL, counts[0]), (S43, counts[1]), (S44, counts[2])]
    exp_f = np.concatenate([np.load(f)['flips'][:n].reshape(-1) for f, n in parts])
    exp_x = np.concatenate([to_sequence(np.load(f)['det_evts'][:n], D, D, 0.004).astype(np.uint8)
                            for f, n in parts])
    n_ok = got['x'].shape[0] == sum(counts) == got['y'].shape[0]
    f_ok = np.array_equal(got['y'][:, 0].astype(np.int8), exp_f.astype(np.int8))
    x_ok = np.array_equal(got['x'], exp_x)
    # each segment must match its own pool
    b = np.cumsum((0,) + counts)
    seg_ok = all(np.array_equal(got['x'][b[i]:b[i + 1]], exp_x[b[i]:b[i + 1]]) for i in range(3))
    ok = n_ok and f_ok and x_ok and seg_ok
    print(f"[multi] {'PASS' if ok else 'FAIL'} order: rows {got['x'].shape[0]:,} = "
          f"{counts[0]:,}+{counts[1]:,}+{counts[2]:,} -> {n_ok}; labels in order FORMAL, 43, 44 "
          f"-> {f_ok}; inputs -> {x_ok}; each segment matches its pool -> {seg_ok}", flush=True)
    full = 15_000_000 + min(5_000_000, np.load(S43)['flips'].shape[0]) + \
        min(30_000_000, np.load(S44)['flips'].shape[0])
    print(f"[multi] full-size count for the 50M run: 15,000,000 + 5,000,000 + 30,000,000 "
          f"available -> {full:,}", flush=True)
    return ok


def dupe(out):
    r1 = run_trainer(base_args(out, 'dupe1', 20_000) + ['--extra-train-pool', S43, S43])
    link = os.path.join(out, 'link_to_seed43.npz')
    for src, dst in ((S43, link), (S43.replace('.npz', '.fingerprint.json'),
                                   link.replace('.npz', '.fingerprint.json'))):
        if not os.path.exists(dst):
            os.symlink(src, dst)
    r2 = run_trainer(base_args(out, 'dupe2', 20_000) + ['--extra-train-pool', S43, link])
    ok1 = isinstance(r1, str) and 'listed twice' in r1
    ok2 = isinstance(r2, str) and 'share gen_seed' in r2
    print(f"[multi] {'PASS' if ok1 and ok2 else 'FAIL'} dupe: same file twice refused -> {ok1} "
          f"({str(r1)[:90]}); same seed via a second path refused -> {ok2} ({str(r2)[:90]})",
          flush=True)
    return ok1 and ok2


if __name__ == '__main__':
    mode, out = sys.argv[1], sys.argv[sys.argv.index('--out') + 1]
    os.makedirs(out, exist_ok=True)
    ok = order(out) if mode == 'order' else dupe(out)
    sys.exit(0 if ok else 1)
