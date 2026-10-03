#!/usr/bin/env python3
# Created: 2026-10-02
# Last modified: 2026-10-02
"""Unit tests for the run-time guards of train_student.py, eval_student_on_tail.py and
partitions.py. CPU only, no TensorFlow, no real pools: every pool is a small file written to
a temporary directory (a FORMAL-named pool holds 19,000,000 zero flips, which compress to a
few kB on disk).

Each guard is tested twice: it raises on a bad configuration and passes on a good one. The
guards in train_student.run() that run before TensorFlow is imported are also tested
through the command line, so the test shows they are wired into the script.

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

N_FORMAL = 19_000_000
SEALED = (17_000_000, 19_000_000)


def write_pool(path, n, fingerprint=None):
    """A pool file with only `flips` (all the guards read), plus an optional fingerprint."""
    np.savez_compressed(path, flips=np.zeros((n, 1), np.int8))
    if fingerprint is not None:
        with open(path.replace('.npz', '.fingerprint.json'), 'w') as fh:
            json.dump(fingerprint, fh)
    return path


class Pools:
    """Fake pools shared by the tests (written once)."""
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


def ns(**kw):
    """train_student arguments used by check_pools_and_partitions; a valid launcher config."""
    a = dict(d=5, p=0.004, rounds=5, pool=Pools.formal, test_pool=None,
             extra_train_pool=[], extra_train_n=[], n_train=15_000_000,
             val_start=15_000_000, val_n=200_000, eval_start=15_200_000, n_test=1_800_000)
    a.update(kw)
    return argparse.Namespace(**a)


def check(**kw):
    a = ns(**kw)
    T.check_pools_and_partitions(a, a.pool)


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
        # eval_start None and no test pool: the last n_test shots = the sealed block.
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
        # Validation and scoring both come from --test-pool. A validation block inside the
        # primary pool's training range is fine there (different file)...
        check(test_pool=Pools.test_pool, val_start=0, val_n=100_000,
              eval_start=None, n_test=200_000)
        # ...but overlapping the test pool's own scored block is not.
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


def run_train(*extra):
    """train_student.py on the command line; returns (exit code, combined output)."""
    base = ['--student', 'gru', '--inputs', 'evts', '--hidden', '--units', '16',
            '--d', '5', '--p', '0.004', '--rounds', '5', '--pool', Pools.formal,
            '--n-train', '1000', '--n-test', '1000', '--seed', '0', '--cpu']
    r = subprocess.run([sys.executable, os.path.join(SRC, 'train_student.py')] + base
                       + list(extra), capture_output=True, text=True, cwd=SRC)
    return r.returncode, r.stdout + r.stderr


class TestTrainCommandLine(unittest.TestCase):
    """Guards that run before TensorFlow is imported, through the real entry point."""

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
        # Passes every guard and prints the block summary; what follows (TensorFlow) is
        # not part of this test, so only the guard output is checked.
        rc, out = run_train('--alpha', '1.0', '--eval-start', '15200000',
                            '--val-start', '15000000', '--val-n', '1000')
        self.assertIn('blocks disjoint: training prefix [0, 1,000)', out)
        self.assertIn('sealed block of data_d5_p0.004_r5_FORMAL.npz: [17,000,000, '
                      '19,000,000) (configs/partitions.yaml; fingerprint agrees)', out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
