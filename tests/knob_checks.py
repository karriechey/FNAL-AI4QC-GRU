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
        prim = lambda d: sorted(f for f in glob.glob(f'{a.pool_dir}/data_d{d}_p0.004_r{d}*.npz')
                                if 'TRAINONLY' not in f)
        self.pool = {d: prim(d)[0] for d in (5, 7, 9) if prim(d)}
        self.ext = sorted(glob.glob(f'{a.pool_dir}/data_d5_p0.004_r5_TRAINONLY*.npz'))
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

    def ema_on(self):
        import h5py
        import numpy as np

        def weights(path):
            out = {}
            with h5py.File(path, 'r') as f:
                f.visititems(lambda k, v: out.__setitem__(k, v[()])
                             if isinstance(v, h5py.Dataset) and 'optimizer' not in k and 'metrics' not in k else None)
            return out

        def equal(a, b):
            return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)

        rc, log, h = self.run_new('ema_on', ['--ema-decay', '0.9'], epochs=2)
        o = f'{self.a.out}/ema_on'
        best_ep = json.load(open(f'{o}/ckpt/t.best.json'))['best_epoch'] if rc == 0 else None
        best, last, end = (weights(f'{o}/ckpt/t.best.weights.h5'),
                           weights(f'{o}/ckpt/t.lastepoch.weights.h5'),
                           weights(f'{o}/runs/t.weights.h5')) if rc == 0 else ({}, {}, {})
        import csv
        row = next(csv.DictReader(open(f'{o}/runs/t.csv'))) if rc == 0 else {}
        # EMA weights differ from the raw weights; when the best epoch is the last one, the
        # saved best equals the end-of-fit weights (Keras copies the EMA into the model at
        # the end of fit), so the checkpoint holds the EMA weights.
        ok = (rc == 0 and not equal(best, last) and not equal(end, last)
              and (best_ep != 2 or equal(best, end))
              and len(h.get('val_loss_ema', [])) == 2
              and h['val_loss_ema'] != h['val_loss']
              and row.get('scored_weights') == 'end_of_fit_ema')
        # One epoch: the best epoch is necessarily the last, so the saved best must equal
        # the end-of-fit (EMA) weights exactly and differ from the raw weights.
        rc1, _, _ = self.run_new('ema_on_1ep', ['--ema-decay', '0.9'], epochs=1)
        o1 = f'{self.a.out}/ema_on_1ep'
        b1, l1, e1 = ((weights(f'{o1}/ckpt/t.best.weights.h5'),
                       weights(f'{o1}/ckpt/t.lastepoch.weights.h5'),
                       weights(f'{o1}/runs/t.weights.h5')) if rc1 == 0 else ({}, {1: 1}, {}))
        ok = ok and rc1 == 0 and equal(b1, e1) and not equal(b1, l1)
        # Training itself is unchanged by EMA: same raw val_loss as without it.
        ok = ok and h.get('val_loss') == getattr(self, 'defaults_hist', h).get('val_loss')
        self.report('ema_on', ok,
                    f"[1 epoch: best==EMA end-of-fit {equal(b1, e1)}, best==raw {equal(b1, l1)}] "
                    f"best epoch {best_ep}; best==raw last-epoch {equal(best, last)}; "
                    f"best==end-of-fit {equal(best, end)}; val_loss raw {h.get('val_loss')} "
                    f"ema {h.get('val_loss_ema')}; scored_weights {row.get('scored_weights')}")

    def warm_start_on(self):
        import h5py
        import numpy as np

        def weights(path):
            out = {}
            with h5py.File(path, 'r') as f:
                f.visititems(lambda k, v: out.__setitem__(k, v[()])
                             if isinstance(v, h5py.Dataset) and 'optimizer' not in k and 'metrics' not in k else None)
            return out

        # Source: d=7 (48 positions), saved with the warmup schedule in its optimizer.
        rs, _, _ = self.run_new('ws_source', ['--lr-warmup-epochs', '0.5'], d=7, epochs=1)
        src = f'{self.a.out}/ws_source/ckpt/t.best.keras'
        # Target: d=9 (80 positions), learning rate ~0 so the loaded values survive the
        # one epoch unchanged to within ~1e-9.
        rc, log, _ = self.run_new('ws_target', ['--warm-start', src, '--lr', '1e-12',
                                                '--lr-end', '1e-13'], d=9, epochs=1)
        loaded = re.findall(r'warm-start: loaded (\S+)', log)
        fresh = re.findall(r'warm-start: fresh  (\S+)', log)
        ws = weights(f'{self.a.out}/ws_source/ckpt/t.best.weights.h5') if rs == 0 else {}
        wt = weights(f'{self.a.out}/ws_target/ckpt/t.lastepoch.weights.h5') if rc == 0 else {}
        close = {k: bool(np.allclose(ws[k], wt[k], atol=1e-8))
                 for k in ws if k in wt and ws[k].shape == wt[k].shape}
        diff_shape = [k for k in ws if k in wt and ws[k].shape != wt[k].shape]
        ok = (rs == 0 and rc == 0 and len(loaded) == 4 and len(fresh) == 1
              and 'kernel' in fresh[0] and 'recurrent' not in fresh[0]
              and close and all(close.values()) and len(diff_shape) == 1)
        self.report('warm_start_on', ok, f"loaded {loaded}; fresh {fresh}; "
                    f"source==target after 1 epoch at lr~0: {close}; shape differs: {diff_shape}")

    def max_params_on(self):
        # u32 at d=5 has 5,505 parameters: a cap of 5,505 passes, 5,504 stops before training.
        rc_ok, log_ok, _ = self.run_new('maxp_ok', ['--max-params', '5505'], epochs=1)
        rc_no, log_no, _ = self.run_new('maxp_no', ['--max-params', '5504'], epochs=1)
        ok = (rc_ok == 0 and rc_no != 0 and 'exceed --max-params 5,504' in log_no
              and 'Epoch 1' not in log_no and '40/40' not in log_no)
        self.report('max_params_on', ok, f"cap 5505 exit {rc_ok}; cap 5504 exit {rc_no}")

    def stacked_on(self):
        # Parameter counts at d=7: one u560 layer and two stacked u280 layers (the ceiling
        # probe arms), and the u280 single layer, which must be unchanged (276,641).
        code = ("import os; os.environ['CUDA_VISIBLE_DEVICES']=''\n"
                "from StudentModels import build_student\n"
                "for u in (280, (280,), 560, (280, 280)):\n"
                "    m = build_student('gru', d=7, rounds=7, hidden=(), units=u)\n"
                "    print(u, m.count_params(), [l.name for l in m.layers])\n")
        r = subprocess.run([self.a.py, '-c', code], cwd=self.new_src, capture_output=True,
                           text=True, env=dict(os.environ, TF_CPP_MIN_LOG_LEVEL='3'))
        counts = dict(re.findall(r'^(\S+(?: \S+)?) (\d+) ', r.stdout, re.M))
        # Train two stacked layers for one epoch, then rebuild and score with the eval script.
        rc, log, _ = self.run_new('stacked_train', ['--units', '16', '16'], epochs=1)
        w = glob.glob(f'{self.a.out}/stacked_train/ckpt/*.best.weights.h5')
        ev = subprocess.run([self.a.py, 'eval_student_on_tail.py', '--student', 'gru', '--hidden',
                             '--units', '16', '16', '--d', '5', '--p', '0.004', '--rounds', '5',
                             '--pool', self.pool[5], '--weights', w[0] if w else 'missing',
                             '--eval-start', '50000', '--n-test', '10000', '--cpu'],
                            cwd=self.new_src, capture_output=True, text=True,
                            env=dict(os.environ, TF_CPP_MIN_LOG_LEVEL='2'))
        ok = (r.returncode == 0 and counts.get('280') == '276641' and counts.get('(280,)') == '276641'
              and counts.get('560') == '1023681' and counts.get('(280, 280)') == '747881'
              and "'gru', 'gru_1'" in r.stdout and rc == 0 and ev.returncode == 0
              and 'student p_L=' in ev.stdout)
        layers = re.findall(r'\[.*\]', r.stdout)[-1:]
        score = (re.findall(r'student p_L=\S+', ev.stdout) or [ev.stderr[-300:]])[0]
        self.report('stacked_on', ok, f"params {counts}; layers: {layers}; "
                    f"train 16x16 exit {rc}, eval exit {ev.returncode} {score}")

    def distill_on(self):
        # Teacher: a d=5 u32 model trained for 2 epochs (the defaults_off run's best.keras).
        # Student: u16, alpha 0 (soft targets only), primary prefix 40,000 + 30,000 extension
        # shots. Every training shot must get a teacher logit (none exactly zero), and the
        # validation and scored blocks too; the guard against extension pools with alpha < 1
        # must not fire with --teacher-model.
        teacher = glob.glob(f'{self.a.out}/defaults_new/ckpt/*.best.keras')
        if not teacher:
            self.defaults_off()
            teacher = glob.glob(f'{self.a.out}/defaults_new/ckpt/*.best.keras')
        rc, log, h = self.run_new('distill', ['--units', '16', '--alpha', '0', '--teacher-model',
                                              teacher[0], '--extra-train-pool', self.ext[0]],
                                  epochs=1)
        n = re.search(r'logits for all ([\d,]+) training shots.*?exactly-zero logits (\d+)', log)
        import csv
        row = (next(csv.DictReader(open(glob.glob(f'{self.a.out}/distill/runs/*.csv')[0])))
               if rc == 0 else {})
        ok = (rc == 0 and n is not None and n.group(1) == '70,000' and n.group(2) == '0'
              and 'teacher logits on the validation block' in log
              and 'agreement vs teacher on the scored block' in log
              and row.get('alpha') == '0.0' and row.get('teacher_cache', '').startswith('model:')
              and row.get('agree_all', '') != '')
        # Same run without --teacher-model must still be refused (extension, alpha < 1).
        rc2, log2, _ = self.run_new('distill_refused', ['--units', '16', '--alpha', '0',
                                                        '--teacher-cache', 'x.npz',
                                                        '--extra-train-pool', self.ext[0]], epochs=1)
        ok = ok and rc2 != 0 and 'no teacher logits' in log2
        lines = [l for l in log.splitlines() if 'teacher' in l][:4]
        self.report('distill_on', ok, f"{lines}; csv alpha={row.get('alpha')} teacher_cache="
                    f"{row.get('teacher_cache')} agree_all={row.get('agree_all')}; "
                    f"refused without model: {rc2 != 0}")

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
    ap.add_argument('checks', nargs='*', default=['defaults_off', 'warmup_on', 'ema_on', 'warm_start_on', 'max_params_on', 'stacked_on', 'distill_on'])
    a = ap.parse_args()
    sys.exit(Checks(a).run(a.checks))
