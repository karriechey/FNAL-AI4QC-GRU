# Created: 2026-09-27
# Last modified: 2026-09-27
"""Sanity check of the d=7 50M training set before training on it (CPU only).

    python3 sanity_d7_50M.py --out <dir>

The 50M set = FORMAL train [0, 15M) + TRAINONLY seed 43 [0, 5M) + TRAINONLY seed 44 [0, 30M),
appended in that order by train_student.py's multi-pool loader.

1 Loader vs direct read. train_student.py is run on the CPU up to the point where it hands
  the training arrays to host_data.make_dataset; the arrays are captured there (training is
  not run). For each part, the captured labels must equal the pool's flips exactly, and the
  captured inputs must equal to_sequence() of the pool's detection events exactly (compared
  in 1M-row chunks).
2 Per-part statistics, read directly from each pool: detection-event density, logical-flip
  rate (label mean), and the MWPM logical error rate on the first 5M shots of each part
  (same decoder and detector error model as every baseline).
  Pass: flip rate and density of each part within 1 % (relative) of the FORMAL part, and the
  MWPM error rate within 3 combined binomial standard errors of the FORMAL part's.
3 50M vs 20M loss curves (informational): training and validation loss of the finished d=7
  reset_after=True runs with 20M and 50M shots (and the old reset_after=False 20M run),
  plotted to loss_curves_20M_vs_50M.png.

Writes sanity_result.json with "passed": true/false and every number used.
"""
import argparse
import glob
import json
import os
import runpy
import sys

import numpy as np

REPO = os.path.expanduser('~/FNAL-AI4QC-GRU')
SRC = os.path.join(REPO, 'src')
sys.path.insert(0, SRC)
POOLS = os.path.expanduser('~/pools_d7_p004_19M')
PARTS = [('FORMAL', f'{POOLS}/data_d7_p0.004_r7_FORMAL.npz', 15_000_000),
         ('seed43', f'{POOLS}/data_d7_p0.004_r7_TRAINONLY_seed43.npz', 5_000_000),
         ('seed44', f'{POOLS}/data_d7_p0.004_r7_TRAINONLY_seed44.npz', 30_000_000)]
CHUNK = 1_000_000
N_MWPM = 5_000_000


class Captured(Exception):
    pass


def capture_loader(out):
    import host_data
    got = {}
    real = host_data.make_dataset

    def capture(tf, x_u8, y=None, batch_size=10_000, shuffle_seed=None, to_gpu=True):
        if shuffle_seed is not None:
            got['x'], got['y'] = x_u8, y
            raise Captured()
        return real(tf, x_u8, y, batch_size, shuffle_seed, to_gpu)
    host_data.make_dataset = capture
    args = ['--student', 'gru', '--inputs', 'evts', '--hidden', '--units', '140',
            '--d', '7', '--p', '0.004', '--rounds', '7', '--pool', PARTS[0][1],
            '--n-train', str(PARTS[0][2]), '--val-start', '15000000', '--val-n', '200000',
            '--eval-start', '15200000', '--n-test', '1800000', '--alpha', '1.0',
            '--temperature', '1.0', '--batch-size', '10000', '--lr', '0.003', '--epochs', '1',
            '--no-early-stopping', '--seed', '0', '--cpu',
            '--extra-train-pool', PARTS[1][1], PARTS[2][1],
            '--extra-train-n', str(PARTS[1][2]), str(PARTS[2][2]),
            '--ckpt-dir', os.path.join(out, 'tmp_ckpt'), '--out-dir', os.path.join(out, 'tmp_runs'),
            '--tag', 'sanity']
    sys.argv = [os.path.join(SRC, 'train_student.py')] + args
    cwd = os.getcwd(); os.chdir(SRC)
    try:
        runpy.run_path(os.path.join(SRC, 'train_student.py'), run_name='__main__')
    except Captured:
        pass
    finally:
        os.chdir(cwd)
    return got['x'], got['y']


def mwpm_ler(det, flips):
    import pymatching
    from circuit_generators import build_circuit
    dem = build_circuit(7, 0.004, 7).detector_error_model(decompose_errors=True)
    m = pymatching.Matching.from_detector_error_model(dem)
    wrong = 0
    for i in range(0, len(det), CHUNK):
        pred = m.decode_batch(det[i:i + CHUNK]).astype(np.int8).reshape(-1)
        wrong += int(np.sum(pred != flips[i:i + CHUNK]))
    return wrong / len(det), wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    from StudentModels import to_sequence
    res = dict(parts={}, loader={})

    x, y = capture_loader(a.out)
    total = sum(n for _, _, n in PARTS)
    res['loader']['rows'] = int(x.shape[0])
    res['loader']['rows_expected'] = total
    ok_all = x.shape[0] == total
    off = 0
    for name, path, n in PARTS:
        z = np.load(path)
        flips = z['flips'][:n].reshape(-1).astype(np.int8)
        det = z['det_evts'][:n]
        lab_ok = bool(np.array_equal(y[off:off + n, 0].astype(np.int8), flips))
        in_ok = True
        for i in range(0, n, CHUNK):
            j = min(i + CHUNK, n)
            if not np.array_equal(x[off + i:off + j], to_sequence(det[i:j], 7, 7, 0.004).astype(np.uint8)):
                in_ok = False
                break
        ler, wrong = mwpm_ler(det[:N_MWPM], flips[:N_MWPM])
        st = dict(range=[off, off + n], labels_match=lab_ok, inputs_match=in_ok,
                  flip_rate=float(flips.mean()), det_density=float(det.mean()),
                  mwpm_ler_first_5M=ler, mwpm_errors_first_5M=wrong)
        res['parts'][name] = st
        ok_all = ok_all and lab_ok and in_ok
        print(f'[sanity] {name}: ' + json.dumps(st), flush=True)
        off += n
        del z, det, flips
    f = res['parts']['FORMAL']
    checks = {}
    for name in ('seed43', 'seed44'):
        p = res['parts'][name]
        rel_flip = abs(p['flip_rate'] - f['flip_rate']) / f['flip_rate']
        rel_den = abs(p['det_density'] - f['det_density']) / f['det_density']
        se = np.sqrt(f['mwpm_ler_first_5M'] * (1 - f['mwpm_ler_first_5M']) / N_MWPM
                     + p['mwpm_ler_first_5M'] * (1 - p['mwpm_ler_first_5M']) / N_MWPM)
        z_ler = abs(p['mwpm_ler_first_5M'] - f['mwpm_ler_first_5M']) / se
        checks[name] = dict(rel_diff_flip_rate=rel_flip, rel_diff_det_density=rel_den,
                            mwpm_ler_diff_in_se=float(z_ler),
                            ok=bool(rel_flip <= 0.01 and rel_den <= 0.01 and z_ler <= 3))
        ok_all = ok_all and checks[name]['ok']
    res['checks_vs_FORMAL'] = checks

    # 3 loss curves (informational)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    Q = os.path.join(REPO, 'experiments', 'queue_ra1_20M')
    runs = {
        'reset_after=True, 20M': glob.glob(f'{Q}/d7_s0_2*/seed0/runs/*.history.json'),
        'reset_after=True, 50M': glob.glob(f'{Q}/d7_s0_50M_*/seed0/runs/*.history.json'),
        'reset_after=False, 20M (old)': glob.glob(os.path.expanduser(
            '~/rcnn_threshold/results/d7_u140_ntr20M_aug_20260821T193446Z/seed0/runs/*.history.json')),
    }
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    res['loss_curves'] = {}
    for (label, files), col in zip(runs.items(), ('#e34948', '#8b1c1b', '#2a78d6')):
        if not files:
            continue
        h = json.load(open(sorted(files)[-1]))
        ep = range(1, len(h['loss']) + 1)
        ax[0].plot(ep, h['loss'], color=col, lw=1.2, label=label)
        ax[1].plot(ep, h['val_loss'], color=col, lw=1.0, label=label)
        res['loss_curves'][label] = dict(final_train_loss=h['loss'][-1],
                                         min_val_loss=min(h['val_loss']),
                                         epoch_min_val_loss=h['val_loss'].index(min(h['val_loss'])) + 1)
    for axx, t in zip(ax, ('training loss', 'validation loss')):
        axx.set_title(f'd=7 u140 seed 0: {t}'); axx.set_xlabel('training epoch')
        axx.set_ylabel('binary cross-entropy loss'); axx.grid(alpha=.3); axx.legend()
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'loss_curves_20M_vs_50M.png'), dpi=130)

    res['passed'] = bool(ok_all)
    with open(os.path.join(a.out, 'sanity_result.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    print(f"[sanity] PASSED={res['passed']} checks={json.dumps(checks)}", flush=True)


if __name__ == '__main__':
    main()
