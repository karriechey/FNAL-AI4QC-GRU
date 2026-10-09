#!/usr/bin/env python3
# Created: 2026-10-09
# Last modified: 2026-10-09
"""Checks of the fresh-shot training input (src/fresh_shots.py,
train_student.py --fresh-shots-per-epoch). CPU only.

  1 layout         fresh_shots._scatter equals StudentModels.to_sequence on the same
                   detection events.
  2 distribution   the detector sampler (fresh shots) and the pools' sampler
                   (generate_pools.sample_pool: measurements, then the converter) give the
                   same logical-flip rate and per-detector rates, d5 p 0.004, 1M shots each:
                   largest |z| over the 120 detectors and the flip rate below 5.
  3 seeds          the same epoch gives the same batches twice; two epochs share no chunk
                   seed and give different batches; check_seeds refuses a range holding a
                   pool seed.
  4 keras epochs   model.fit over FreshShots.dataset samples epochs 0, 1, 2 in order, one per
                   fit epoch; first_epoch=2 starts at epoch 2 with the same batches.
  6 bounded        with a slow consumer, never more than in_flight chunks are requested
                   ahead of the one consumed (an unbounded queue held ~40 GB in the first
                   full run).
  5 train_student  --fresh-shots-per-epoch runs end to end (2 epochs), twice with identical
                   per-epoch losses, writes fresh_shots = shots x epochs in its CSV, and is
                   refused with --n-train > 0 or alpha < 1.

    python3 tests/fresh_shots_test.py --py <python with TF 2.15> --pool-dir <dir> --out <dir>
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(REPO, 'src'))
results = []


def report(name, ok, detail):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--py', default=sys.executable)
    ap.add_argument('--pool-dir', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import fresh_shots as F
    from circuit_generators import build_circuit
    from generate_pools import sample_pool
    from StudentModels import to_sequence, detector_sequence_layout
    d, p = 5, 0.004

    # 1 ------------------------------------------------------------------------------
    circ = build_circuit(d, p, d)
    det, _ = circ.compile_detector_sampler(seed=5).sample(20_000, separate_observables=True)
    _, _, imap = detector_sequence_layout(d, d, p)
    ok = np.array_equal(F._scatter(det.astype(np.uint8), imap).astype(np.float32),
                        to_sequence(det.astype(np.uint8), d, d, p))
    report('layout', ok, f'20,000 shots, shape {F._scatter(det.astype(np.uint8), imap).shape}')

    # 2 ------------------------------------------------------------------------------
    n = 1_000_000
    det_f, obs_f = circ.compile_detector_sampler(seed=11).sample(n, separate_observables=True)
    _, det_p, flips_p = sample_pool(circ, n, 12)
    rf, rp = det_f.mean(0), det_p.astype(np.float64).mean(0)
    pooled = (rf + rp) / 2
    z_det = np.abs(rf - rp) / np.sqrt(np.maximum(pooled * (1 - pooled) * 2 / n, 1e-30))
    ff, fp = obs_f.mean(), flips_p.astype(np.float64).mean()
    pf = (ff + fp) / 2
    z_flip = abs(ff - fp) / np.sqrt(pf * (1 - pf) * 2 / n)
    report('distribution', z_det.max() < 5 and z_flip < 5,
           f'flip rate {ff:.5f} vs pool path {fp:.5f} (|z| {z_flip:.2f}); detector rates '
           f'max |z| {z_det.max():.2f} over {len(z_det)} detectors')

    # 3 ------------------------------------------------------------------------------
    fs = F.FreshShots(d, p, d, 40_000, 1_000, chunk=10_000, workers=2)
    e3a = [x.copy() for x, _ in fs.batches(3)]
    e3b = [x.copy() for x, _ in fs.batches(3)]
    e4 = [x.copy() for x, _ in fs.batches(4)]
    same = all(np.array_equal(u, v) for u, v in zip(e3a, e3b)) and len(e3a) == 40
    differ = not any(np.array_equal(u, v) for u, v in zip(e3a, e4))
    s3 = {F.chunk_seed(3, c) for c in range(fs.chunks_per_epoch)}
    s4 = {F.chunk_seed(4, c) for c in range(fs.chunks_per_epoch)}
    try:
        F.check_seeds(10, 4, seed_base=40)       # range [40, 9,000,044) holds pool seed 42
        refused = False
    except ValueError:
        refused = True
    report('seeds', same and differ and not (s3 & s4) and refused,
           f'epoch 3 repeated identical: {same}; epoch 4 differs: {differ}; shared chunk '
           f'seeds: {len(s3 & s4)}; range holding seed 42 refused: {refused}')

    # 4 ------------------------------------------------------------------------------
    import tensorflow as tf
    asked = []

    class Rec(F.FreshShots):
        def batches(self, epoch):
            asked.append(epoch)
            return super().batches(epoch)

    rs = Rec(d, p, d, 4_000, 1_000, chunk=2_000, workers=2)
    T, P = rs.shape
    m = tf.keras.Sequential([tf.keras.layers.Input((T, P)), tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(1)])
    m.compile('adam', loss=lambda y, z: tf.reduce_mean(tf.square(z[:, 0] - y[:, 0])))
    m.fit(rs.dataset(tf), epochs=3, verbose=0)
    order = list(asked)
    asked.clear()
    m.fit(rs.dataset(tf, first_epoch=2), epochs=3, initial_epoch=2, verbose=0)
    resumed = list(asked)
    first2 = [x.numpy() for x, _ in rs.dataset(tf, first_epoch=2, to_gpu=False).take(2)]
    direct2 = [x for x, _ in F.FreshShots.batches(rs, 2)][:2]
    ok = (order == [0, 1, 2] and resumed == [2] and
          all(np.array_equal(u.astype(np.uint8), v) for u, v in zip(first2, direct2)))
    report('keras_epochs', ok, f'fit 3 epochs sampled epochs {order}; fit resumed at epoch 2 '
           f'sampled {resumed}; its batches equal epoch 2 sampled directly')
    rs.close()

    # 6 ------------------------------------------------------------------------------
    import time
    submitted, consumed, ahead = [0], [0], []

    def counting(*args, **kw):          # uses `real`, bound below to the new pool's method
        submitted[0] += 1
        ahead.append(submitted[0] - consumed[0])
        return real(*args, **kw)

    fs.close()
    fs = F.FreshShots(d, p, d, 200_000, 1_000, chunk=10_000, workers=2, in_flight=3)
    real = fs.pool.apply_async
    fs.pool.apply_async = counting
    for i, _ in enumerate(fs.batches(7)):
        if i % 10 == 0:                     # 10 batches per chunk: one chunk consumed
            consumed[0] += 1
            time.sleep(0.05)
    report('bounded', max(ahead) <= 3 + 1 and submitted[0] == fs.chunks_per_epoch,
           f'{submitted[0]} chunks requested, at most {max(ahead)} ahead of consumption '
           f'(window 3)')
    fs.close()

    # 5 ------------------------------------------------------------------------------
    pool = os.path.join(a.pool_dir, 'data_d5_p0.004_r5_rowtest.npz')
    common = ['--student', 'gru', '--inputs', 'evts', '--hidden', '--units', '16',
              '--d', '5', '--p', '0.004', '--rounds', '5', '--pool', pool,
              '--val-start', '40000', '--val-n', '10000', '--eval-start', '50000',
              '--n-test', '10000', '--temperature', '1.0', '--batch-size', '1000',
              '--lr', '0.003', '--lr-schedule', 'cosine', '--lr-end', '3e-5',
              '--global-clipnorm', '0.64', '--epochs', '2', '--no-early-stopping',
              '--seed', '0', '--require-determinism', '--cpu', '--tag', 't', '--run-tag', 't',
              '--fresh-shots-per-epoch', '20000', '--fresh-workers', '2']
    env = dict(os.environ, TF_DETERMINISTIC_OPS='1', TF_CUDNN_DETERMINISTIC='1')
    src = os.path.join(REPO, 'src')
    hist, rows, rcs = [], [], []
    for i in range(2):
        o = os.path.join(a.out, f'train{i}')
        r = subprocess.run([a.py, 'train_student.py'] + common
                           + ['--n-train', '0', '--alpha', '1.0',
                              '--ckpt-dir', o + '/ckpt', '--out-dir', o + '/runs'],
                           cwd=src, env=env, capture_output=True, text=True)
        open(os.path.join(a.out, f'train{i}.log'), 'w').write(r.stdout + r.stderr)
        rcs.append(r.returncode)
        hp = glob.glob(o + '/runs/*.history.json')
        hist.append(json.load(open(hp[0])) if hp else {})
        cp = glob.glob(o + '/runs/*.csv')
        rows.append(next(csv.DictReader(open(cp[0]))) if cp else {})
    same = hist[0].get('loss') and hist[0].get('loss') == hist[1].get('loss') \
        and hist[0].get('val_ler') == hist[1].get('val_ler')
    refusals = []
    for extra in (['--n-train', '40000', '--alpha', '1.0'], ['--n-train', '0', '--alpha', '0.5',
                                                         '--teacher-model', '/nonexistent.keras']):
        r = subprocess.run([a.py, 'train_student.py'] + common + extra
                           + ['--ckpt-dir', a.out + '/x', '--out-dir', a.out + '/x'],
                           cwd=src, env=env, capture_output=True, text=True)
        refusals.append(r.returncode != 0 and 'fresh-shots-per-epoch needs' in r.stdout + r.stderr)
    ok = (rcs == [0, 0] and same and rows[0].get('fresh_shots') == '40000'
          and rows[0].get('train_pool', '').startswith('fresh_stim_seeds_') and all(refusals))
    report('train_student', ok, f"exit {rcs}; loss {hist[0].get('loss')} twice equal: {bool(same)}; "
           f"fresh_shots {rows[0].get('fresh_shots')}; train_pool {rows[0].get('train_pool')}; "
           f"refusals {refusals}")

    print(f"\n{sum(results)}/{len(results)} checks passed", flush=True)
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
