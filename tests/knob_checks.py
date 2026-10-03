#!/usr/bin/env python3
# Created: 2026-10-03
# Last modified: 2026-10-03
"""Cheap CPU checks for the training settings added for the d=9 queue (warmup, EMA,
warm-start, parameter cap). No GPU, small pools, a few epochs each.

Two kinds of check:
  defaults_off   the current train_student.py with none of the new flags gives the same
                 parameter count and the same per-epoch loss / val_loss as the code at
                 --base-ref (determinism flags on, CPU)
  <knob>_on      the flag does what it says, read from the log and the outputs

Pools: any pool files data_d{d}_p0.004_r{d}*.npz with a fingerprint, at least 60,000 shots
(blocks used: train [0, 40000), validation [40000, 50000), scored [50000, 60000)).

    python3 tests/knob_checks.py --py <python with TF 2.15> --pool-dir <dir> \\
        --base-ref 0372b35 --out <dir> [checks ...]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def common(pool, d, units=32, epochs=2):
    return ['--student', 'gru', '--inputs', 'evts', '--hidden', '--units', str(units),
            '--d', str(d), '--p', '0.004', '--rounds', str(d), '--pool', pool,
            '--n-train', '40000', '--val-start', '40000', '--val-n', '10000',
            '--eval-start', '50000', '--n-test', '10000', '--alpha', '1.0',
            '--temperature', '1.0', '--batch-size', '1000', '--lr', '0.003',
            '--lr-schedule', 'cosine', '--lr-end', '3e-5', '--global-clipnorm', '0.64',
            '--epochs', str(epochs), '--no-early-stopping', '--seed', '0',
            '--require-determinism', '--cpu', '--tag', 't', '--run-tag', 't']


def train(py, src, out, args):
    """Run train_student.py from `src` with outputs in `out`; return (log text, history)."""
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, TF_DETERMINISTIC_OPS='1', TF_CUDNN_DETERMINISTIC='1',
               TF_CPP_MIN_LOG_LEVEL='2', CUDA_VISIBLE_DEVICES='')
    r = subprocess.run([py, 'train_student.py'] + args
                       + ['--ckpt-dir', f'{out}/ckpt', '--out-dir', f'{out}/runs'],
                       cwd=src, env=env, capture_output=True, text=True)
    log = r.stdout + r.stderr
    open(f'{out}/run.log', 'w').write(log)
    hist = {}
    hp = glob.glob(f'{out}/runs/*.history.json')
    if hp:
        hist = json.load(open(hp[0]))
    return r.returncode, log, hist


def params(log):
    m = re.search(r'params=([\d,]+)', log)
    return int(m.group(1).replace(',', '')) if m else None


def same_history(h1, h2, keys=('loss', 'val_loss')):
    return all(h1.get(k) == h2.get(k) and h1.get(k) for k in keys)


class Checks:
    def __init__(self, a):
        self.a = a
        self.pool = {d: sorted(glob.glob(f'{a.pool_dir}/data_d{d}_p0.004_r{d}*.npz'))[0]
                     for d in (5, 7, 9)
                     if glob.glob(f'{a.pool_dir}/data_d{d}_p0.004_r{d}*.npz')}
        self.base_src = os.path.join(a.out, 'base', 'src')
        if not os.path.exists(self.base_src):
            os.makedirs(os.path.join(a.out, 'base'), exist_ok=True)
            archive = subprocess.run(['git', '-C', REPO, 'archive', a.base_ref, 'src',
                                      'configs'], capture_output=True, check=True).stdout
            subprocess.run(['tar', '-x', '-C', os.path.join(a.out, 'base')], input=archive,
                           check=True)
        self.new_src = os.path.join(REPO, 'src')
        self.results = []

    def report(self, name, ok, detail):
        self.results.append((name, ok))
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)

    def run_new(self, name, extra, d=5, **kw):
        return train(self.a.py, self.new_src, f'{self.a.out}/{name}',
                     common(self.pool[d], d, **kw) + extra)

    def defaults_off(self):
        args = common(self.pool[5], 5)
        rb, lb, hb = train(self.a.py, self.base_src, f'{self.a.out}/defaults_base', args)
        rn, ln, hn = train(self.a.py, self.new_src, f'{self.a.out}/defaults_new', args)
        ok = rb == 0 and rn == 0 and params(lb) == params(ln) and same_history(hb, hn)
        self.report('defaults_off', ok,
                    f"params {params(lb)} vs {params(ln)}; loss {hb.get('loss')} vs "
                    f"{hn.get('loss')}; val_loss {hb.get('val_loss')} vs {hn.get('val_loss')}")
        self.defaults_hist = hn

    def warmup_on(self):
        rc, log, h = self.run_new('warmup_on', ['--lr-warmup-epochs', '1'], epochs=3)
        m = dict((int(k.replace(',', '')), float(v))
                 for k, v in re.findall(r'lr\[([\d,]+)\]=([0-9.e+-]+)', log))
        # 40 updates per epoch, warmup 40 updates: lr[0] = 3e-3/40, lr[39] = 3e-3,
        # lr[40] = cosine start = 3e-3, then decreasing to 3e-5 at the last update.
        ok = (rc == 0 and abs(m.get(0, 0) - 3e-3 / 40) < 1e-9 and abs(m.get(39, 0) - 3e-3) < 1e-9
              and abs(m.get(40, 0) - 3e-3) < 1e-9 and m.get(41, 1) < 3e-3
              and abs(m.get(119, 0) - 3e-5) < 2e-6 and h.get('lr', [None])[0] is not None)
        self.report('warmup_on', ok, f"lr marks {m}; history lr per epoch {h.get('lr')}")

    def run(self, names):
        for n in names:
            getattr(self, n)()
        bad = [n for n, ok in self.results if not ok]
        print(f"\n{len(self.results) - len(bad)}/{len(self.results)} checks passed"
              + (f"; FAILED: {bad}" if bad else ''), flush=True)
        return 0 if not bad else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--py', default=sys.executable)
    ap.add_argument('--pool-dir', required=True)
    ap.add_argument('--base-ref', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('checks', nargs='*', default=['defaults_off', 'warmup_on'])
    a = ap.parse_args()
    sys.exit(Checks(a).run(a.checks))
