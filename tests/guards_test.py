#!/usr/bin/env python3
# Created: 2026-10-02
# Last modified: 2026-10-06 (held-out files sealed whole; seed reuse caught)

"""Unit tests for the run-time guards of train_student.py, eval_student_on_tail.py and partitions.py.

Includes the sealed-block rule, the sealed-access log and the CSV writer. CPU only, small fake
pools. Each guard is tested on a bad and a good configuration.

python3 tests/guards_test.py        (from the repo root)
"""
import argparse
import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC)
import partitions as P                      # noqa: E402
import train_student as T                   # noqa: E402  (no TensorFlow at import)
import eval_student_on_tail as E            # noqa: E402  (no TensorFlow or stim at import)

N_FORMAL = 19_000_000
SEALED = (17_000_000, 19_000_000)


def write_pool(path, n, fingerprint=None):
    # pool with only `flips`, plus an optional fingerprint
    np.savez_compressed(path, flips=np.zeros((n, 1), np.int8))
    if fingerprint is not None:
        with open(path.replace('.npz', '.fingerprint.json'), 'w') as fh:
            json.dump(fingerprint, fh)
    return path


class Pools:
    tmp = tempfile.mkdtemp(prefix='guards_test_')
    atexit.register(shutil.rmtree, tmp, True)
    formal = write_pool(os.path.join(tmp, 'data_d5_p0.004_r5_FORMAL.npz'), N_FORMAL,
                        dict(d=5, p=0.004, rounds=5, gen_seed=42,
                             partition_val=[15_000_000, 17_000_000],      # the stale field
                             partition_test_SEALED=list(SEALED)))
    formal_bad_fp = write_pool(os.path.join(tmp, 'data_d5_p0.004_r5_FORMAL_badfp.npz'),
                               N_FORMAL, dict(d=5, p=0.004, rounds=5, gen_seed=42,
                                              partition_test_SEALED=[18_000_000, 19_000_000]))
    formal_short = write_pool(os.path.join(tmp, 'data_d5_p0.004_r5_FORMAL_short.npz'), 1000,
                              dict(d=5, p=0.004, rounds=5, gen_seed=42))
    trainonly = write_pool(os.path.join(tmp, 'data_d5_p0.004_r5_TRAINONLY_seed43.npz'), 50_000,
                           dict(d=5, p=0.004, rounds=5, gen_seed=43, purpose='training_only',
                                partition_test_SEALED=None))
    other = write_pool(os.path.join(tmp, 'data_d5_p0.004_r5.npz'), 100_000)   # name only
    test_pool = write_pool(os.path.join(tmp, 'data_d5_p0.004_r5_fresh_tail.npz'), 400_000)
    noname = write_pool(os.path.join(tmp, 'pool_without_geometry.npz'), 1000)
    wrong_p_ext = write_pool(os.path.join(tmp, 'data_d5_p0.005_r5_TRAINONLY_x.npz'), 1000)
    short_p_name = write_pool(os.path.join(tmp, 'data_d7_p004_r7_x.npz'), 1000)
    weights = os.path.join(tmp, 'w.best.weights.h5')
    weights_other = os.path.join(tmp, 'other.weights.h5')
    for _w, _b in ((weights, b'weights one'), (weights_other, b'weights two')):
        with open(_w, 'wb') as _fh:
            _fh.write(_b)
    # test copy of the sealed-access log
    sealed_log = os.path.join(tmp, 'sealed_access.tsv')
    os.environ['SEALED_ACCESS_LOG'] = sealed_log


def ns(**kw):
    # valid train_student arguments for check_pools_and_partitions
    a = dict(d=5, p=0.004, rounds=5, pool=Pools.formal, test_pool=None,
             extra_train_pool=[], extra_train_n=[], n_train=15_000_000,
             val_start=15_000_000, val_n=200_000, eval_start=15_200_000, n_test=1_800_000)
    a.update(kw)
    return argparse.Namespace(**a)


def check(**kw):
    a = ns(**kw)
    T.check_pools_and_partitions(a, a.pool)


class TestHeldOut(unittest.TestCase):
    def test_heldout_file_sealed_whole_and_refused(self):
        ho = write_pool(os.path.join(Pools.tmp, 'data_d5_p0.004_r5_HELDOUT_seed7001.npz'), 5000,
                        dict(d=5, p=0.004, rounds=5, gen_seed=7001, purpose='held_out'))
        self.assertEqual(P.sealed_block(ho, 5000)[0], (0, 5000))
        with self.assertRaises(SystemExit):
            E.check_eval_block(eval_args(pool=ho, eval_start=0, n_test=100), ())

    def test_seed_reuse_caught(self):
        cfg = P.load()
        cfg = dict(cfg, held_out=list(cfg.get('held_out') or []) + [dict(gen_seed=42)])
        with self.assertRaises(ValueError):
            P.check(cfg)
        self.assertIn(7001, P.used_seeds())


class TestDisjoint(unittest.TestCase):
    def test_overlap_same_pool_raises(self):
        with self.assertRaises(ValueError):
            P.check_disjoint([('a', 'x.npz', 0, 10), ('b', 'x.npz', 9, 20)])

    def test_touching_and_other_pool_pass(self):
        self.assertTrue(P.check_disjoint([('a', 'x.npz', 0, 10), ('b', 'x.npz', 10, 20),
                                          ('c', 'y.npz', 0, 20), ('empty', 'x.npz', 5, 5)]))


class TestSealedBlock(unittest.TestCase):
    def test_formal_from_yaml_with_fingerprint_agreeing(self):
        s, why = P.sealed_block(Pools.formal, N_FORMAL)
        self.assertEqual(s, SEALED)
        self.assertIn('fingerprint agrees', why)

    def test_fingerprint_disagreeing_raises(self):
        with self.assertRaises(ValueError):
            P.sealed_block(Pools.formal_bad_fp, N_FORMAL)

    def test_formal_wrong_size_raises(self):
        with self.assertRaises(ValueError):
            P.sealed_block(Pools.formal_short, 1000)

    def test_trainonly_and_unmarked_have_none(self):
        self.assertIsNone(P.sealed_block(Pools.trainonly, 50_000)[0])
        s, why = P.sealed_block(Pools.other, 100_000)
        self.assertIsNone(s)
        self.assertIn('not marked FORMAL', why)


class TestGeometry(unittest.TestCase):
    def test_fingerprint_mismatch_raises(self):
        with self.assertRaises(ValueError):
            P.check_pool_geometry(Pools.formal, 7, 0.004, 7)
        with self.assertRaises(ValueError):
            P.check_pool_geometry(Pools.formal, 5, 0.005, 5)

    def test_name_mismatch_raises(self):
        with self.assertRaises(ValueError):
            P.check_pool_geometry(Pools.other, 5, 0.004, 3)

    def test_p_in_both_name_styles(self):
        self.assertEqual(P.p_from_name('0.004'), 0.004)
        self.assertEqual(P.p_from_name('004'), 0.004)
        self.assertEqual(P.p_from_name('010'), 0.010)
        self.assertEqual(P.check_pool_geometry(Pools.short_p_name, 7, 0.004, 7), 'file name')
        with self.assertRaises(ValueError):
            P.check_pool_geometry(Pools.short_p_name, 7, 0.005, 7)

    def test_match_and_missing(self):
        self.assertEqual(P.check_pool_geometry(Pools.formal, 5, 0.004, 5), 'fingerprint')
        self.assertEqual(P.check_pool_geometry(Pools.other, 5, 0.004, 5), 'file name')
        self.assertIsNone(P.check_pool_geometry(Pools.noname, 5, 0.004, 5))


class TestTrainPartitions(unittest.TestCase):
    def test_launcher_config_passes(self):
        check()
        check(extra_train_pool=[Pools.trainonly], extra_train_n=[50_000])

    def test_val_equal_to_scored_raises(self):
        with self.assertRaises(SystemExit):
            check(val_start=15_200_000, val_n=1_800_000)

    def test_val_overlapping_scored_raises(self):
        with self.assertRaises(SystemExit):
            check(val_start=15_100_000, val_n=200_000)

    def test_val_overlapping_train_raises(self):
        with self.assertRaises(SystemExit):
            check(val_start=14_900_000, val_n=200_000)

    def test_scored_overlapping_train_raises(self):
        with self.assertRaises(SystemExit):
            check(n_train=15_500_000, val_start=15_500_000, val_n=100_000,
                  eval_start=15_000_000, n_test=100_000)

    def test_scored_into_sealed_raises(self):
        with self.assertRaises(SystemExit):
            check(eval_start=16_000_000, n_test=1_800_000)

    def test_default_tail_is_sealed_raises(self):
        # no eval_start and no test pool: the last n_test shots are the sealed block
        with self.assertRaises(SystemExit):
            check(eval_start=None)

    def test_val_in_sealed_raises(self):
        with self.assertRaises(SystemExit):
            check(val_start=17_000_000, val_n=200_000)

    def test_geometry_mismatch_raises(self):
        with self.assertRaises(SystemExit):
            check(d=7, rounds=7)
        with self.assertRaises(SystemExit):
            check(extra_train_pool=[Pools.wrong_p_ext], extra_train_n=[1000])

    def test_test_pool_val_checked_against_its_own_scored_block(self):
        # with --test-pool, validation inside the primary training range is fine...
        check(test_pool=Pools.test_pool, val_start=0, val_n=100_000,
              eval_start=None, n_test=200_000)
        # ...but not overlapping the test pool's scored block
        with self.assertRaises(SystemExit):
            check(test_pool=Pools.test_pool, val_start=250_000, val_n=100_000,
                  eval_start=None, n_test=200_000)


class TestHidden(unittest.TestCase):
    def test_defaults_and_explicit(self):
        self.assertEqual(T.resolve_hidden('gru', None), ())
        self.assertEqual(T.resolve_hidden('mlp', None), (128, 128))
        self.assertEqual(T.resolve_hidden('gru', []), ())          # bare --hidden
        self.assertEqual(T.resolve_hidden('mlp', [160, 160]), (160, 160))
        self.assertEqual(T.resolve_hidden('gru', [64]), (64,))


def eval_args(**kw):
    a = dict(pool=Pools.formal, weights=Pools.weights, student='gru', units=[280], d=5,
             p=0.004, rounds=5, eval_start=15_200_000, n_test=1_800_000,
             score_sealed=False, score_sealed_again=False)
    a.update(kw)
    return argparse.Namespace(**a)


def log_rows():
    import csv
    if not os.path.exists(Pools.sealed_log):
        return []
    with open(Pools.sealed_log, newline='') as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


class TestEvalSealed(unittest.TestCase):
    def setUp(self):
        if os.path.exists(Pools.sealed_log):
            os.remove(Pools.sealed_log)

    def test_eval_block_passes_and_logs_nothing(self):
        self.assertEqual(E.check_eval_block(eval_args(), ()), (15_200_000, 17_000_000))
        self.assertEqual(log_rows(), [])

    def test_sealed_refused_without_flag(self):
        with self.assertRaises(SystemExit):
            E.check_eval_block(eval_args(eval_start=17_000_000, n_test=2_000_000), ())
        with self.assertRaises(SystemExit):                       # partial overlap
            E.check_eval_block(eval_args(eval_start=16_900_000, n_test=200_000), ())
        self.assertEqual(log_rows(), [])

    def test_again_needs_score_sealed(self):
        with self.assertRaises(SystemExit):
            E.check_eval_block(eval_args(score_sealed_again=True), ())

    def test_score_sealed_logs_once_then_refuses_then_repeat(self):
        sealed = dict(eval_start=17_000_000, n_test=2_000_000, score_sealed=True)
        self.assertEqual(E.check_eval_block(eval_args(**sealed), ()), (17_000_000, 19_000_000))
        rows = log_rows()
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r['weights_sha256'], E.sha256_of(Pools.weights))
        self.assertEqual((r['d'], r['p'], r['rounds'], r['units']), ('5', '0.004', '5', '280'))
        self.assertEqual((r['block_start'], r['block_stop'], r['repeat']),
                         ('17000000', '19000000', '0'))
        self.assertTrue(r['utc'].endswith('Z'))
        # same weights again: refused, nothing logged
        with self.assertRaises(SystemExit):
            E.check_eval_block(eval_args(**sealed), ())
        self.assertEqual(len(log_rows()), 1)
        # other weights: allowed
        E.check_eval_block(eval_args(weights=Pools.weights_other, **sealed), ())
        self.assertEqual(len(log_rows()), 2)
        # same weights with --score-sealed-again: allowed, marked as a repeat
        E.check_eval_block(eval_args(score_sealed_again=True, **sealed), ())
        rows = log_rows()
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[-1]['repeat'], '1')

    def test_command_line_refuses_sealed(self):
        r = subprocess.run([sys.executable, os.path.join(SRC, 'eval_student_on_tail.py'),
                            '--weights', Pools.weights, '--student', 'gru', '--hidden',
                            '--units', '280', '--d', '5', '--p', '0.004', '--rounds', '5',
                            '--pool', Pools.formal, '--eval-start', '17000000',
                            '--n-test', '1000', '--cpu'],
                           capture_output=True, text=True, cwd=SRC)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('needs --score-sealed', r.stdout + r.stderr)
        self.assertEqual(log_rows(), [])


class TestEvalCsv(unittest.TestCase):
    def row(self, tag):
        return [tag] + ['x'] * (len(E.CSV_COLS) - 1)

    def read(self, path):
        import csv
        with open(path, newline='') as fh:
            return list(csv.reader(fh))

    def test_new_file_full_header(self):
        out = os.path.join(Pools.tmp, 'eval_new.csv')
        self.assertEqual(E.append_eval_row(out, self.row('a')), out)
        self.assertEqual(self.read(out), [E.CSV_COLS, self.row('a')])

    def test_old_header_goes_to_v2(self):
        out = os.path.join(Pools.tmp, 'eval_best_ckpt.csv')
        old_header = E.CSV_COLS[:-2]
        with open(out, 'w') as fh:
            fh.write(','.join(old_header) + '\n' + ','.join(['old'] * len(old_header)) + '\n')
        before = open(out).read()
        v2 = os.path.join(Pools.tmp, 'eval_best_ckpt.v2.csv')
        self.assertEqual(E.append_eval_row(out, self.row('a')), v2)
        self.assertEqual(E.append_eval_row(out, self.row('b')), v2)
        self.assertEqual(open(out).read(), before)                 # old file untouched
        self.assertEqual(self.read(v2), [E.CSV_COLS, self.row('a'), self.row('b')])

    def test_v2_with_other_header_refused(self):
        out = os.path.join(Pools.tmp, 'eval_x.csv')
        for path in (out, os.path.join(Pools.tmp, 'eval_x.v2.csv')):
            with open(path, 'w') as fh:
                fh.write('weights,student\n')
        with self.assertRaises(SystemExit):
            E.append_eval_row(out, self.row('a'))


def run_train(*extra):
    # (exit code, output) of train_student.py on the command line
    base = ['--student', 'gru', '--inputs', 'evts', '--hidden', '--units', '16',
            '--d', '5', '--p', '0.004', '--rounds', '5', '--pool', Pools.formal,
            '--n-train', '1000', '--n-test', '1000', '--seed', '0', '--cpu']
    r = subprocess.run([sys.executable, os.path.join(SRC, 'train_student.py')] + base
                       + list(extra), capture_output=True, text=True, cwd=SRC)
    return r.returncode, r.stdout + r.stderr


class TestTrainCommandLine(unittest.TestCase):
    # guards before the TensorFlow import, through the real entry point

    def test_alpha_below_one_with_extension_refused(self):
        rc, out = run_train('--alpha', '0.5', '--teacher-cache', 'unused.npz',
                            '--eval-start', '15200000',
                            '--extra-train-pool', Pools.trainonly)
        self.assertNotEqual(rc, 0)
        self.assertIn('--extra-train-pool with alpha=0.5', out)

    def test_no_eval_start_no_test_pool_refused(self):
        rc, out = run_train('--alpha', '1.0')
        self.assertNotEqual(rc, 0)
        self.assertIn('give --eval-start', out)

    def test_scored_in_sealed_refused(self):
        rc, out = run_train('--alpha', '1.0', '--eval-start', '17000000')
        self.assertNotEqual(rc, 0)
        self.assertIn('overlaps sealed block', out)

    def test_geometry_refused(self):
        rc, out = run_train('--alpha', '1.0', '--eval-start', '15200000', '--rounds', '3')
        self.assertNotEqual(rc, 0)
        self.assertIn('fingerprint says d=5 p=0.004 r=5', out)

    def test_valid_config_passes_the_guards(self):
        # passes every guard; only the guard output is checked
        rc, out = run_train('--alpha', '1.0', '--eval-start', '15200000',
                            '--val-start', '15000000', '--val-n', '1000')
        self.assertIn('blocks disjoint: training prefix [0, 1,000)', out)
        self.assertIn('sealed block of data_d5_p0.004_r5_FORMAL.npz: [17,000,000, '
                      '19,000,000) (configs/partitions.yaml; fingerprint agrees)', out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
