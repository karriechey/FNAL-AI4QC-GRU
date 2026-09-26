# Created: 2026-09-25
# Last modified: 2026-09-25
"""A5 equivalence tests: does the new repo reproduce the old pipeline exactly?

Run on grace1 from the repo root, with the old repo checked out at d5ee86f:

    TF_CPP_MIN_LOG_LEVEL=3 ~/tfenv/bin/python tests/a5_equivalence.py

Six checks, each printed as PASS or FAIL, plus a JSON report written to
experiments/a5_equivalence/<UTC timestamp>/report.json:

  1 circuits      old eval_on_tail.build_circuit and generate_pools.build_circuit vs the
                  new circuit_generators.build_circuit and generate_pools.build_circuit:
                  identical str(circuit) and detector error model at d=5,7,9, p=0.004, r=d.
  2 partitions    (a) configs/partitions.yaml ranges equal the literals in the old launcher
                  and old distillation trainer; (b) each FORMAL / TRAINONLY pool's flips
                  SHA-256 and gen_seed match the yaml and the pool's fingerprint;
                  (c) the new sampler with gen_seed 42 regenerates the first 1,000,000
                  shots of each FORMAL pool bit-for-bit; (d) SHA-256 of the val and eval
                  blocks (and of the sealed flips, as a hash only) are recorded.
  3 disjointness  train/val/eval/sealed ranges disjoint and tiling [0, 19M); training
                  defaults end at or before the val block; FORMAL and TRAINONLY seeds differ.
                  Compared by seed and shot index, never by syndrome contents.
  4 mwpm          old and new mwpm_on_block.py give the same failure count on 100,000
                  eval-block shots at d=5.
  5 scatter       old and new to_sequence give identical (r+1, d^2-1) tensors on 10,000
                  eval-block shots at d=5,7,9, and every detector maps exactly once.
  6 model         the d=5 u140 run-of-record checkpoint gives identical logits in the old
                  and new code on 10,000 eval-block shots.

The sealed block's contents are never decoded, scored or passed to a model. Test 2 only
hashes its flips column, as the step allows.
"""
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, SRC)

import partitions as P  # noqa: E402  (needs SRC on the path)

P_PHYS = 0.004
DISTANCES = (5, 7, 9)
N_SCATTER = 10_000
N_MWPM = 100_000
N_REGEN = 1_000_000          # = generate_pools.CHUNK, the first sampler call
PY = sys.executable

RESULTS = {}
EVAL_SLICES = {}          # d -> first N_SCATTER eval-block rows of det_evts, read once


def record(name, ok, **info):
    RESULTS[name] = dict(passed=bool(ok), **info)
    print(f"[A5] {'PASS' if ok else 'FAIL'}  {name}  "
          + '  '.join(f'{k}={v}' for k, v in info.items() if not isinstance(v, (dict, list))),
          flush=True)


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_txt(t):
    return hashlib.sha256(t.encode()).hexdigest()


def old_side(*args):
    """Run tests/old_side.py against the old repo in a separate process."""
    cmd = [PY, os.path.join(ROOT, 'tests', 'old_side.py'), '--old-repo', OLD, *map(str, args)]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------------------
paths = P._load('paths.yaml')
OLD = os.path.expanduser(os.environ.get('A5_OLD_REPO', paths['old_repo']))   # env override for dry runs
cfg = P.load()
formal = cfg['formal']
TMP = tempfile.mkdtemp(prefix='a5_')


def old_head():
    return subprocess.check_output(['git', '-C', OLD, 'rev-parse', 'HEAD']).decode().strip()


def old_dirty():
    return subprocess.check_output(['git', '-C', OLD, 'status', '--porcelain',
                                    '--untracked-files=no']).decode().strip()


head = old_head()
print(f"[A5] old repo {OLD} at {head[:12]}", flush=True)
if not head.startswith('d5ee86f') or old_dirty():
    raise SystemExit(f"[A5] old repo must be a clean checkout of d5ee86f (HEAD={head[:12]}, "
                     f"modified tracked files: {bool(old_dirty())}). STOP.")


# 1 circuits ------------------------------------------------------------------------------
def test_circuits():
    import circuit_generators as cg
    import generate_pools as gp
    out = os.path.join(TMP, 'old_circuits.json')
    old_side('--task', 'circuits', '--out', out)
    old = json.load(open(out))
    ok, info = True, {}
    for d in DISTANCES:
        new = {'circuit_generators': cg.build_circuit(d, P_PHYS, d),
               'generate_pools': gp.build_circuit(d, P_PHYS)}
        shas = set()
        for name, c in new.items():
            shas.add((sha_txt(str(c)), sha_txt(str(c.detector_error_model(decompose_errors=True)))))
        for name in ('eval_on_tail', 'generate_pools'):
            o = old[f'{name}_d{d}']
            shas.add((o['circuit_sha'], o['dem_sha']))
        info[f'd{d}_distinct_versions'] = len(shas)
        ok &= len(shas) == 1
    record('1_circuits', ok, **info)


# 2 partitions ----------------------------------------------------------------------------
def old_literals():
    """Partition literals as written in the old launcher and old distillation trainer."""
    sh = open(os.path.join(OLD, 'gh200_run_student_scaling.sh')).read()
    py = open(os.path.join(OLD, 'train_gru_distillation_20m.py')).read()
    g = lambda pat, txt: int(re.search(pat, txt).group(1).replace('_', ''))
    launcher = dict(val=(g(r'VAL_START="\$\{VAL_START:-(\d+)\}"', sh),
                         g(r'VAL_START="\$\{VAL_START:-(\d+)\}"', sh)
                         + g(r'VAL_N="\$\{VAL_N:-(\d+)\}"', sh)),
                    eval=(g(r'EVAL_START="\$\{EVAL_START:-(\d+)\}"', sh),
                          g(r'EVAL_START="\$\{EVAL_START:-(\d+)\}"', sh)
                          + g(r'EVAL_N="\$\{EVAL_N:-(\d+)\}"', sh)),
                    sealed_start=g(r'SEALED_START="\$\{SEALED_START:-(\d+)\}"', sh))
    distill = dict(train_stop=g(r"'--main-n-train', type=int, default=([\d_]+)", py),
                   val=(g(r"'--val-start', type=int, default=([\d_]+)", py),
                        g(r"'--val-start', type=int, default=([\d_]+)", py)
                        + g(r"'--val-n', type=int, default=([\d_]+)", py)),
                   eval=(g(r"'--eval-start', type=int, default=([\d_]+)", py),
                         g(r"'--eval-start', type=int, default=([\d_]+)", py)
                         + g(r"'--eval-n', type=int, default=([\d_]+)", py)),
                   sealed_start=g(r"'--sealed-start', type=int, default=([\d_]+)", py))
    return launcher, distill


def test_partitions():
    from circuit_generators import build_circuit
    from generate_pools import sample_pool
    ok = True
    tr, va, ev, se = (P.block(b, cfg) for b in P.BLOCKS)
    launcher, distill = old_literals()
    lit_ok = (launcher['val'] == va and launcher['eval'] == ev and launcher['sealed_start'] == se[0]
              and distill['train_stop'] == tr[1] and distill['val'] == va
              and distill['eval'] == ev and distill['sealed_start'] == se[0])
    record('2a_ranges_match_old_literals', lit_ok, yaml=dict(train=tr, val=va, eval=ev, sealed=se),
           old_launcher=launcher, old_distill=distill)
    ok &= lit_ok

    hashes = {}
    for kind in ('formal', 'trainonly'):
        for d, want_sha in cfg[kind]['flips_sha256'].items():
            path = P.pool_path(d, kind)
            fp = json.load(open(path.replace('.npz', '.fingerprint.json')))
            z = np.load(path)
            flips = z['flips']
            got = sha_arr(flips)
            this = (got == want_sha == fp['flips_sha256']
                    and fp['gen_seed'] == cfg[kind]['gen_seed']
                    and flips.shape[0] == cfg[kind]['n_total'] == fp['n_total'])
            fp_parts = {k: v for k, v in fp.items() if 'partition' in k}
            record(f'2b_pool_identity_{kind}_d{d}', this, flips_sha=got[:16],
                   gen_seed=fp['gen_seed'], n=flips.shape[0], fingerprint_partitions=fp_parts)
            ok &= this
            if kind != 'formal':
                continue
            det = z['det_evts']
            # (c) regenerate the first chunk with the new sampler and the yaml seed.
            m, e, f = sample_pool(build_circuit(d, P_PHYS, d), N_REGEN, formal['gen_seed'])
            regen = np.array_equal(e, det[:N_REGEN]) and np.array_equal(f, flips[:N_REGEN])
            record(f'2c_regenerate_first_{N_REGEN}_d{d}', regen)
            ok &= regen
            # (d) block hashes under the yaml ranges (= old literals if 2a passed).
            hashes[d] = dict(val_det=sha_arr(det[va[0]:va[1]]), val_flips=sha_arr(flips[va[0]:va[1]]),
                             eval_det=sha_arr(det[ev[0]:ev[1]]), eval_flips=sha_arr(flips[ev[0]:ev[1]]),
                             sealed_flips_hash_only=sha_arr(flips[se[0]:se[1]]))
            e0 = ev[0]
            EVAL_SLICES[d] = np.array(det[e0:e0 + N_SCATTER])
            del det, z
    RESULTS['2d_block_hashes'] = hashes
    record('2_partitions', ok)


# 3 disjointness --------------------------------------------------------------------------
def test_disjointness():
    ok = P.check(cfg)
    tr, va, ev, se = (P.block(b, cfg) for b in P.BLOCKS)
    sh = open(os.path.join(SRC, 'gh200_run_student_scaling.sh')).read()
    ntrain_default = int(re.search(r'NTRAIN="\$\{NTRAIN:-(\d+)\}"', sh).group(1))
    seeds = {'formal': cfg['formal']['gen_seed'], 'trainonly': cfg['trainonly']['gen_seed']}
    blocks = [tr, va, ev, se]
    pair_ok = all(a[1] <= b[0] for a, b in zip(blocks, blocks[1:]))
    ok = ok and pair_ok and ntrain_default <= va[0] and tr[1] <= va[0] \
        and seeds['formal'] != seeds['trainonly']
    record('3_disjointness', ok, launcher_ntrain_default=ntrain_default,
           val_start=va[0], seeds=seeds)


# 4 mwpm ----------------------------------------------------------------------------------
def test_mwpm():
    d, (e0, _) = 5, P.block('eval', cfg)
    pool = P.pool_path(d, 'formal')
    res = {}
    for side, script, cwd in (('old', os.path.join(OLD, 'mwpm_on_block.py'), OLD),
                              ('new', os.path.join(SRC, 'mwpm_on_block.py'), SRC)):
        out = os.path.join(TMP, f'mwpm_{side}.json')
        subprocess.run([PY, script, '--pool', pool, '--d', str(d), '--p', str(P_PHYS),
                        '--rounds', str(d), '--start', str(e0), '--n', str(N_MWPM),
                        '--out-json', out], check=True, cwd=cwd)
        res[side] = json.load(open(out))['residual_errors']
    record('4_mwpm_d5', res['old'] == res['new'], old_failures=res['old'],
           new_failures=res['new'], shots=N_MWPM, start=e0)


def eval_slice(d):
    """First N_SCATTER eval-block rows for distance d, plus a file copy for old_side.py."""
    if d not in EVAL_SLICES:                     # test 2 normally fills this
        e0 = P.block('eval', cfg)[0]
        EVAL_SLICES[d] = np.array(np.load(P.pool_path(d, 'formal'))['det_evts'][e0:e0 + N_SCATTER])
    path = os.path.join(TMP, f'eval_slice_d{d}.npz')
    if not os.path.exists(path):
        np.savez(path, det=EVAL_SLICES[d])
    return EVAL_SLICES[d], path


# 5 scatter -------------------------------------------------------------------------------
def test_scatter():
    from StudentModels import to_sequence, detector_sequence_layout
    ok = True
    for d in DISTANCES:
        out = os.path.join(TMP, f'scatter_d{d}.npz')
        det, slice_path = eval_slice(d)
        old_side('--task', 'scatter', '--slice', slice_path, '--d', d, '--out', out)
        o = np.load(out)
        new = to_sequence(det, d, d, P_PHYS)
        n_t, n_pos, im = detector_sequence_layout(d, d, P_PHYS)
        used = np.sort(im[im >= 0])
        once = np.array_equal(used, np.arange(det.shape[1]))
        shape_ok = new.shape[1:] == (d + 1, d * d - 1)
        this = np.array_equal(o['seq'], new) and np.array_equal(o['index_map'], im) and once and shape_ok
        record(f'5_scatter_d{d}', this, shape=new.shape[1:], detectors=det.shape[1],
               each_once=once)
        ok &= this
    record('5_scatter', ok)


# 6 model ---------------------------------------------------------------------------------
def test_model():
    from StudentModels import to_sequence, build_gru_student
    d = 5
    run = paths['runs_of_record']['d5_u140_10M']
    w = os.path.join(os.path.expanduser(paths['results_root']), run, 'seed0', 'ckpt',
                     'gru_d5_p004_u140_hard_seed0_ntr10000000_b10000.best.weights.h5')
    det, slice_path = eval_slice(d)
    out = os.path.join(TMP, 'logits_old.npz')
    old_side('--task', 'logits', '--slice', slice_path, '--d', d, '--weights', w, '--out', out)
    o = np.load(out)
    model = build_gru_student(d=d, rounds=d, units=140, p=P_PHYS)
    model.load_weights(w)
    new = model.predict(to_sequence(det, d, d, P_PHYS), batch_size=10000, verbose=0).reshape(-1)
    diff = float(np.max(np.abs(new - o['logits'])))
    ok = np.array_equal(new, o['logits']) and int(o['n_params']) == model.count_params() == 69441
    record('6_model_logits_d5_u140', ok, max_abs_diff=diff, params=model.count_params(),
           checkpoint=run + '/seed0')


# -----------------------------------------------------------------------------------------
if __name__ == '__main__':
    stamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    outdir = os.path.join(ROOT, 'experiments', 'a5_equivalence', stamp)
    os.makedirs(outdir, exist_ok=True)
    for t in (test_circuits, test_partitions, test_disjointness, test_mwpm, test_scatter,
              test_model):
        try:
            t()
        except Exception as exc:          # a crash is a failure, not a skip
            record(t.__name__, False, error=f'{type(exc).__name__}: {exc}')
    import platform, stim, pymatching, tensorflow as tf
    RESULTS['_env'] = dict(host=platform.node(), machine=platform.machine(),
                           python=platform.python_version(), stim=stim.__version__,
                           pymatching=pymatching.__version__, tensorflow=tf.__version__,
                           old_repo_head=head, new_repo_head=subprocess.check_output(
                               ['git', '-C', ROOT, 'rev-parse', 'HEAD']).decode().strip())
    with open(os.path.join(outdir, 'report.json'), 'w') as fh:
        json.dump(RESULTS, fh, indent=1, default=str)
    fails = [k for k, v in RESULTS.items() if isinstance(v, dict) and v.get('passed') is False]
    print(f"[A5] report -> {outdir}/report.json")
    print(f"[A5] {'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)
