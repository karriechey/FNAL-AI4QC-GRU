# Created: 2026-09-25
# Last modified: 2026-10-02 (sealed_block, check_disjoint, check_pool_geometry: run-time guards used by train_student.py and eval_student_on_tail.py). Earlier 2026-09-25
"""Read configs/partitions.yaml and configs/paths.yaml.

This is the only code that knows the partition ranges. Python scripts call the functions
below; shell launchers call the command-line form, which prints shell assignments:

    eval "$(python partitions.py --shell --d 7)"
    # -> TRAIN_START=0 TRAIN_STOP=15000000 VAL_START=15000000 VAL_N=200000
    #    EVAL_START=15200000 EVAL_N=1800000 SEALED_START=17000000 SEALED_STOP=19000000
    #    POOL=/home/.../data_d7_p0.004_r7_FORMAL.npz FORMAL_GEN_SEED=42 ...

Loading also checks that the four FORMAL ranges are disjoint and tile [0, n_total), so a
bad edit to the YAML fails immediately instead of producing overlapping blocks.

Run-time guards (2026-10-02), used by train_student.py and eval_student_on_tail.py:
  sealed_block(pool, n)      the sealed block of one pool, from this YAML; the pool's
                             .fingerprint.json partition_test_SEALED is a cross-check only
  check_disjoint(blocks)     no two (pool, start, stop) blocks of the same pool overlap
  check_pool_geometry(...)   the pool's d, p, rounds equal the ones a script was given
They raise ValueError; the scripts turn that into SystemExit with their own log prefix.
"""
import argparse
import json
import os
import re

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs')
BLOCKS = ('train', 'val', 'eval', 'sealed')


def _load(name):
    with open(os.path.join(CONFIG_DIR, name)) as fh:
        return yaml.safe_load(fh)


def load():
    """Return the parsed partitions.yaml after checking it."""
    cfg = _load('partitions.yaml')
    check(cfg)
    return cfg


def check(cfg):
    """Raise unless train/val/eval/sealed are disjoint and cover [0, n_total) in order,
    and the TRAINONLY seed differs from the FORMAL seed."""
    f = cfg['formal']
    rng = [tuple(f['ranges'][b]) for b in BLOCKS]
    for (a0, a1), name in zip(rng, BLOCKS):
        if not a0 < a1:
            raise ValueError(f"partition {name} [{a0}, {a1}) is empty or reversed")
    if rng[0][0] != 0 or rng[-1][1] != f['n_total']:
        raise ValueError(f"partitions do not span [0, {f['n_total']})")
    for (a0, a1), (b0, b1), na, nb in zip(rng, rng[1:], BLOCKS, BLOCKS[1:]):
        if a1 != b0:
            raise ValueError(f"{na} ends at {a1} but {nb} starts at {b0} (gap or overlap)")
    if cfg['trainonly']['gen_seed'] == f['gen_seed']:
        raise ValueError("TRAINONLY gen_seed equals FORMAL gen_seed; the extension would "
                         "reproduce FORMAL shots")
    return True


def block(name, cfg=None):
    """(start, stop) half-open shot indices of a FORMAL block."""
    cfg = cfg or load()
    a, b = cfg['formal']['ranges'][name]
    return int(a), int(b)


def pool_path(d, kind='formal'):
    """Path of the FORMAL or TRAINONLY pool for distance d, from paths.yaml."""
    tmpl = _load('paths.yaml')['pools'][kind]
    return os.path.expanduser(tmpl.format(d=d))


def read_fingerprint(pool):
    """The pool's .fingerprint.json as a dict, or None when there is none."""
    fp = pool.replace('.npz', '.fingerprint.json')
    if not os.path.exists(fp):
        return None
    with open(fp) as fh:
        return json.load(fh)


def sealed_block(pool, n_pool, cfg=None):
    """(start, stop) of the sealed block of `pool`, or None, plus a one-line reason.

    configs/partitions.yaml is the source of truth. It defines the sealed block of every
    FORMAL pool (formal.ranges.sealed). A pool is FORMAL when its name contains FORMAL, its
    fingerprint says purpose 'formal', its fingerprint carries a partition_test_SEALED, or
    its generation seed is the FORMAL seed of the YAML. For such a pool the shot count must
    equal formal.n_total, and the fingerprint's partition_test_SEALED, when present, must
    agree with the YAML; either disagreement raises, since then the YAML does not describe
    this file. The fingerprint's other partition fields are not used (the p=0.004 FORMAL
    fingerprints still carry the older merged validation block [15M, 17M)).

    Training-only pools have no sealed block. Any other pool (no FORMAL marker) returns
    None with a reason the caller prints as a warning.
    """
    cfg = cfg or load()
    fp = read_fingerprint(pool) or {}
    name = os.path.basename(pool)
    if fp.get('purpose') == 'training_only' or 'TRAINONLY' in name:
        return None, 'training-only pool: no sealed block'
    formal = ('FORMAL' in name or fp.get('purpose') == 'formal'
              or fp.get('partition_test_SEALED') is not None
              or fp.get('gen_seed') == cfg['formal']['gen_seed'])
    if not formal:
        return None, (f'{name} is not marked FORMAL (name, fingerprint purpose, sealed '
                      f'field or gen_seed); no sealed block is known for it')
    sealed = block('sealed', cfg)
    n_total = int(cfg['formal']['n_total'])
    if int(n_pool) != n_total:
        raise ValueError(f'{name} is a FORMAL pool with {int(n_pool):,} shots, but '
                         f'configs/partitions.yaml defines FORMAL pools of {n_total:,}; '
                         f'its sealed block cannot be located. STOP.')
    fps = fp.get('partition_test_SEALED')
    if fps is not None and (int(fps[0]), int(fps[1])) != sealed:
        raise ValueError(f'{name}: fingerprint partition_test_SEALED {list(fps)} disagrees '
                         f'with configs/partitions.yaml sealed {list(sealed)}. STOP.')
    return sealed, ('configs/partitions.yaml'
                    + ('; fingerprint agrees' if fps is not None else
                       '; fingerprint has no sealed field'))


def check_disjoint(blocks):
    """Raise if two blocks of the same pool overlap.

    blocks: (role, pool_path, start, stop), half-open shot indices. Pools are compared by
    absolute path, so blocks of two different files never conflict (their shots come from
    different generation seeds, which train_student.py checks separately). Empty blocks
    are skipped.
    """
    bl = [(role, os.path.abspath(pool), int(a), int(b)) for role, pool, a, b in blocks
          if int(b) > int(a)]
    for i in range(len(bl)):
        for j in range(i + 1, len(bl)):
            r1, p1, a1, b1 = bl[i]
            r2, p2, a2, b2 = bl[j]
            if p1 == p2 and a1 < b2 and a2 < b1:
                raise ValueError(f'{r1} [{a1:,}, {b1:,}) overlaps {r2} [{a2:,}, {b2:,}) '
                                 f'in {os.path.basename(p1)}. STOP.')
    return True


_NAME_RE = re.compile(r'data_d(\d+)_p([0-9.]+)_r(\d+)')


def check_pool_geometry(pool, d, p, rounds):
    """Raise if the pool's d, p, rounds differ from the given ones; return a source note.

    Reads the pool's .fingerprint.json; falls back to the data_d{d}_p{p}_r{r} pattern of
    the file name. Returns None when neither exists, for the caller to warn.
    """
    fp = read_fingerprint(pool)
    name = os.path.basename(pool)
    if fp is not None and all(k in fp for k in ('d', 'p', 'rounds')):
        have, src = (int(fp['d']), float(fp['p']), int(fp['rounds'])), 'fingerprint'
    else:
        m = _NAME_RE.search(name)
        if not m:
            return None
        have, src = (int(m.group(1)), float(m.group(2)), int(m.group(3))), 'file name'
    want = (int(d), float(p), int(rounds))
    if have[0] != want[0] or abs(have[1] - want[1]) > 1e-9 or have[2] != want[2]:
        raise ValueError(f'{name}: {src} says d={have[0]} p={have[1]} r={have[2]}, the '
                         f'arguments say d={want[0]} p={want[1]} r={want[2]}. STOP.')
    return src


def shell_assignments(d):
    cfg = load()
    tr, va, ev, se = (block(b, cfg) for b in BLOCKS)
    out = dict(TRAIN_START=tr[0], TRAIN_STOP=tr[1],
               VAL_START=va[0], VAL_N=va[1] - va[0],
               EVAL_START=ev[0], EVAL_N=ev[1] - ev[0],
               SEALED_START=se[0], SEALED_STOP=se[1],
               FORMAL_GEN_SEED=cfg['formal']['gen_seed'],
               FORMAL_FLIPS_SHA=cfg['formal']['flips_sha256'][d],
               POOL=pool_path(d, 'formal'))
    return ' '.join(f'{k}={v}' for k, v in out.items())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--shell', action='store_true', help='print shell assignments')
    ap.add_argument('--d', type=int, required=True)
    a = ap.parse_args()
    print(shell_assignments(a.d) if a.shell else load())
