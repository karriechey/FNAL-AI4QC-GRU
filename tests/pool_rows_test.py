#!/usr/bin/env python3
# Created: 2026-10-06
# Last modified: 2026-10-06
"""Row-range pool reading (src/pool_rows.py): same data as before, and nothing outside the
requested blocks is read. CPU only, small pools.

Pools (built by the caller in --pool-dir): a primary pool of 80,000 shots whose rows
[60,000, 80,000) stand in for a sealed block; a full-format and a compact training-only
pool of 30,000 shots each.

Checks
  1 pool_rows.rows / n_rows / scalar equal np.load slicing, for an uncompressed and a
    compressed copy (no TensorFlow).
  2 train_student.py, code before the change (--base-ref) vs now, same arguments:
    train [0, 40,000) + 20,000 full-format + 20,000 compact extension shots, validation
    [40,000, 50,000), scored [50,000, 60,000). Per-epoch loss, val_loss, val p_L (val_ler)
    and the scored p_L must be identical (identical losses mean identical batches).
  3 eval_student_on_tail.py and mwpm_on_block.py, before vs now, on [50,000, 60,000):
    identical p_L, MWPM p_L and McNemar counts.
  4 With POOL_ROWS_TRACE, every row range read by the new code in 2 and 3 lies inside
    the requested blocks; in particular no row >= 60,000 of the primary pool is read.

    python3 tests/pool_rows_test.py --py <python with TF 2.15> --pool-dir <dir> \\
        --base-ref 90724d4 --out <dir>
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
SEALED_START = 60_000
results = []


def report(name, ok, detail):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--py', default=sys.executable)
    ap.add_argument('--pool-dir', required=True)
    ap.add_argument('--base-ref', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prim = os.path.join(a.pool_dir, 'data_d5_p0.004_r5_rowtest.npz')
    ext_full = os.path.join(a.pool_dir, 'data_d5_p0.004_r5_TRAINONLY_seed22.npz')
    ext_cmp = os.path.join(a.pool_dir, 'data_d5_p0.004_r5_TRAINONLY_seed23.npz')
    new_src = os.path.join(REPO, 'src')
    sys.path.insert(0, new_src)
    import pool_rows as PR

    # 1 ------------------------------------------------------------------------------
    z = np.load(prim)
    comp = os.path.join(a.out, 'compressed.npz')
    np.savez_compressed(comp, **{k: z[k] for k in z.files})
    ok = True
    for path in (prim, comp):
        for key in z.files:
            ok &= PR.n_rows(path, key) == z[key].shape[0]
            for lo, hi in ((0, 1), (0, 40_000), (40_000, 50_000), (50_000, 60_000), (79_999, 80_000)):
                ok &= bool(np.array_equal(PR.rows(path, key, lo, hi), z[key][lo:hi]))
    zc = np.load(ext_cmp)
    ok &= int(PR.scalar(ext_cmp, 'n_det')) == int(zc['n_det'])
    ok &= bool(np.array_equal(PR.rows(ext_cmp, 'det_evts_packed', 5, 777), zc['det_evts_packed'][5:777]))
    report('rows_equal_np_load', ok, 'every key, 5 ranges, stored and compressed copies; compact n_det and packed rows')

    # base code ---------------------------------------------------------------------------
    base = os.path.join(a.out, 'base')
    if not os.path.exists(base):
        os.makedirs(base)
        arch = subprocess.run(['git', '-C', REPO, 'archive', a.base_ref, 'src', 'configs'],
                              capture_output=True, check=True).stdout
        subprocess.run(['tar', '-x', '-C', base], input=arch, check=True)
    env0 = dict(os.environ, TF_DETERMINISTIC_OPS='1', TF_CUDNN_DETERMINISTIC='1',
                TF_CPP_MIN_LOG_LEVEL='2', CUDA_VISIBLE_DEVICES='')

    def run(src, script, args, tag, trace=None):
        env = dict(env0)
        if trace:
            env['POOL_ROWS_TRACE'] = trace
        r = subprocess.run([a.py, script] + args, cwd=src, env=env, capture_output=True, text=True)
        open(os.path.join(a.out, tag + '.log'), 'w').write(r.stdout + r.stderr)
        return r

    # 2 ------------------------------------------------------------------------------
    common = ['--student', 'gru', '--inputs', 'evts', '--hidden', '--units', '32',
              '--d', '5', '--p', '0.004', '--rounds', '5', '--pool', prim,
              '--extra-train-pool', ext_full, ext_cmp, '--extra-train-n', '20000', '20000',
              '--n-train', '40000', '--val-start', '40000', '--val-n', '10000',
              '--eval-start', '50000', '--n-test', '10000', '--alpha', '1.0',
              '--temperature', '1.0', '--batch-size', '1000', '--lr', '0.003',
              '--lr-schedule', 'cosine', '--lr-end', '3e-5', '--global-clipnorm', '0.64',
              '--epochs', '2', '--no-early-stopping', '--seed', '0', '--require-determinism',
              '--cpu', '--tag', 't', '--run-tag', 't']
    hist, row, rc = {}, {}, {}
    trace = os.path.join(a.out, 'trace.txt')
    if os.path.exists(trace):
        os.remove(trace)
    for v, src in (('base', os.path.join(base, 'src')), ('new', new_src)):
        o = os.path.join(a.out, 'train_' + v)
        r = run(src, 'train_student.py', common + ['--ckpt-dir', o + '/ckpt', '--out-dir', o + '/runs'],
                'train_' + v, trace if v == 'new' else None)
        rc[v] = r.returncode
        hp = glob.glob(o + '/runs/*.history.json')
        hist[v] = json.load(open(hp[0])) if hp else {}
        cp = glob.glob(o + '/runs/*.csv')
        row[v] = next(csv.DictReader(open(cp[0]))) if cp else {}
    keys = ('loss', 'val_loss', 'val_ler', 'val_errors')
    same = all(hist['base'].get(k) == hist['new'].get(k) and hist['new'].get(k) for k in keys)
    ok = rc['base'] == 0 and rc['new'] == 0 and same and row['base'].get('p_L') == row['new'].get('p_L')
    report('train_identical', ok, f"exit {rc}; loss {hist['new'].get('loss')} vs base "
           f"{hist['base'].get('loss')}; val_ler {hist['new'].get('val_ler')}; scored p_L "
           f"{row['new'].get('p_L')} vs {row['base'].get('p_L')}")

    # 3 ------------------------------------------------------------------------------
    w = glob.glob(os.path.join(a.out, 'train_new', 'ckpt', '*.best.weights.h5'))[0]
    ev = {}
    for v, src in (('base', os.path.join(base, 'src')), ('new', new_src)):
        csvp = os.path.join(a.out, f'eval_{v}.csv')
        if os.path.exists(csvp):
            os.remove(csvp)
        r = run(src, 'eval_student_on_tail.py',
                ['--weights', w, '--student', 'gru', '--hidden', '--units', '32', '--d', '5',
                 '--p', '0.004', '--rounds', '5', '--pool', prim, '--eval-start', '50000',
                 '--n-test', '10000', '--batch-size', '1000', '--cpu', '--out-csv', csvp],
                'eval_' + v, trace if v == 'new' else None)
        e = next(csv.DictReader(open(csvp))) if os.path.exists(csvp) else {}
        r2 = run(src, 'mwpm_on_block.py',
                 ['--pool', prim, '--d', '5', '--p', '0.004', '--rounds', '5', '--start', '50000',
                  '--n', '10000', '--out-json', os.path.join(a.out, f'mwpm_{v}.json')],
                 'mwpm_' + v, trace if v == 'new' else None)
        mj = os.path.join(a.out, f'mwpm_{v}.json')
        m = json.load(open(mj)) if os.path.exists(mj) else {}
        ev[v] = (r.returncode, r2.returncode, {k: e.get(k) for k in ('p_L', 'mwpm_p_L', 'student_only',
                 'mwpm_only', 'n_discordant')}, m.get('mwpm_p_L'), m.get('residual_errors'))
    ok = ev['base'] == ev['new'] and ev['new'][0] == 0 and ev['new'][1] == 0 and ev['new'][2]['p_L']
    report('eval_and_mwpm_identical', ok, f"new {ev['new']}; base {ev['base']}")

    # 4 ------------------------------------------------------------------------------
    reads = [l.split() for l in open(trace)] if os.path.exists(trace) else []
    allowed = {os.path.abspath(prim): [(0, 40_000), (40_000, 50_000), (50_000, 60_000)],
               os.path.abspath(ext_full): [(0, 20_000)], os.path.abspath(ext_cmp): [(0, 20_000)]}
    bad = [rd for rd in reads
           if not any(lo <= int(rd[2]) and int(rd[3]) <= hi for lo, hi in allowed.get(rd[0], []))]
    max_prim = max((int(rd[3]) for rd in reads if rd[0] == os.path.abspath(prim)), default=0)
    report('reads_inside_blocks', bool(reads) and not bad and max_prim <= SEALED_START,
           f"{len(reads)} row reads traced; highest primary row read {max_prim} "
           f"(stand-in sealed block starts at {SEALED_START}); outside the blocks: {bad[:5]}")

    print(f"\n{sum(results)}/{len(results)} checks passed", flush=True)
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
