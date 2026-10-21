# Created: 2026-09-25
# Last modified: 2026-10-18 (fresh_seed_ranges).
# Earlier 2026-10-06 (used_seeds, held-out files sealed whole, spent_blocks, checks over the complete pool table).
# Earlier 2026-10-02 (sealed_block, check_disjoint, check_pool_geometry: run-time guards used by train_student.py and eval_student_on_tail.py; p in file names read in both the p0.004 and p004 styles; sealed-access log).
# Earlier 2026-09-25

"""Read configs/partitions.yaml and configs/paths.yaml;

Shell launchers use the command-line form:

    eval "$(python partitions.py --shell --d 7)"
    # -> TRAIN_START=0 TRAIN_STOP=15000000 VAL_START=15000000 VAL_N=200000 ... POOL=... FORMAL_GEN_SEED=42

Guards raise ValueError. Every scoring of a sealed block is logged in configs/sealed_access.tsv.
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
    cfg = _load('partitions.yaml')
    check(cfg)
    return cfg


def check(cfg):
    # train/val/eval/sealed disjoint and tiling [0, n_total); TRAINONLY seed != FORMAL seed
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
    # within one (d, p) every pool has its own seed; a held-out block's seed is used nowhere else
    seen = {}
    for e in cfg.get('pools', []):
        key = (int(e['d']), float(e['p']), int(e['gen_seed']))
        if key in seen:
            raise ValueError(f"two pools with d={key[0]} p={key[1]} gen_seed={key[2]}")
        seen[key] = e
    others = {int(e['gen_seed']) for e in cfg.get('pools', [])} | \
             {int(k) for k in (cfg.get('seeds_used_elsewhere') or {})}
    ho = [int(e['gen_seed']) for e in (cfg.get('held_out') or [])]
    if len(set(ho)) != len(ho) or set(ho) & others:
        raise ValueError(f"held-out seeds {ho} are reused (other seeds in use: {sorted(others)})")
    spans = sorted(fresh_seed_ranges(cfg))
    for (a0, a1, _), (b0, b1, _) in zip(spans, spans[1:]):
        if b0 < a1:
            raise ValueError(f"fresh seed ranges [{a0}, {a1}) and [{b0}, {b1}) overlap")
    return True


def fresh_seed_ranges(cfg=None):
    from fresh_shots import SEED_EPOCH_STRIDE
    cfg = cfg if cfg is not None else load()
    return [(int(e['base']), int(e['base']) + int(e['max_epochs']) * SEED_EPOCH_STRIDE,
             int(e['max_epochs'])) for e in (cfg.get('fresh_seed_ranges') or [])]


def used_seeds(cfg=None):
    # every seed used by a pool or held-out block, or listed as used elsewhere
    cfg = cfg or load()
    return ({int(e['gen_seed']) for e in cfg.get('pools', [])}
            | {int(e['gen_seed']) for e in (cfg.get('held_out') or [])}
            | {int(k) for k in (cfg.get('seeds_used_elsewhere') or {})})


def spent_blocks(pool, cfg=None):
    # [(start, stop, note)] already read by a decoder
    cfg = cfg or load()
    ap = os.path.abspath(pool)
    return [(int(b['range'][0]), int(b['range'][1]), f"spent {b['spent']} by {b['by']}; {b['use']}")
            for b in (cfg.get('spent_blocks') or [])
            if os.path.abspath(os.path.expanduser(b['pool'])) == ap]


def block(name, cfg=None):
    # (start, stop) of a FORMAL block
    cfg = cfg or load()
    a, b = cfg['formal']['ranges'][name]
    return int(a), int(b)


def pool_path(d, kind='formal'):
    tmpl = _load('paths.yaml')['pools'][kind]
    return os.path.expanduser(tmpl.format(d=d))


def read_fingerprint(pool):
    fp = pool.replace('.npz', '.fingerprint.json')
    if not os.path.exists(fp):
        return None
    with open(fp) as fh:
        return json.load(fh)


def sealed_block(pool, n_pool, cfg=None):
    # (start, stop) of the pool's sealed block or None, plus a reason. The YAML is the source
    # of truth; a FORMAL pool's size and fingerprint sealed range must agree with it.
    cfg = cfg or load()
    fp = read_fingerprint(pool) or {}
    name = os.path.basename(pool)
    # a held-out block is its own file, sealed whole
    ho = {os.path.abspath(os.path.expanduser(e['path'])) for e in (cfg.get('held_out') or [])}
    if fp.get('purpose') == 'held_out' or 'HELDOUT' in name or os.path.abspath(pool) in ho:
        return (0, int(n_pool)), 'held-out block: the whole file is sealed'
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
    # blocks: (role, pool_path, start, stop); only blocks of the same file are compared
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


def p_from_name(text):
    # 'p0.004' (pool files) or 'p004' (folders, run tags; thousandths)
    return float(text) if '.' in text else int(text) / 1000.0


def check_pool_geometry(pool, d, p, rounds):
    # from .fingerprint.json, else the data_d{d}_p{p}_r{r} file name; None if neither
    fp = read_fingerprint(pool)
    name = os.path.basename(pool)
    if fp is not None and all(k in fp for k in ('d', 'p', 'rounds')):
        have, src = (int(fp['d']), float(fp['p']), int(fp['rounds'])), 'fingerprint'
    else:
        m = _NAME_RE.search(name)
        if not m:
            return None
        have, src = (int(m.group(1)), p_from_name(m.group(2)), int(m.group(3))), 'file name'
    want = (int(d), float(p), int(rounds))
    if have[0] != want[0] or abs(have[1] - want[1]) > 1e-9 or have[2] != want[2]:
        raise ValueError(f'{name}: {src} says d={have[0]} p={have[1]} r={have[2]}, the '
                         f'arguments say d={want[0]} p={want[1]} r={want[2]}. STOP.')
    return src


SEALED_LOG_COLS = ['utc', 'weights_sha256', 'weights_path', 'student', 'units', 'hidden',
                   'd', 'p', 'rounds', 'pool', 'block_start', 'block_stop', 'repeat']


def sealed_log_path():
    # SEALED_ACCESS_LOG overrides the path (tests only)
    return os.environ.get('SEALED_ACCESS_LOG') or os.path.join(CONFIG_DIR, 'sealed_access.tsv')


def sealed_accesses(weights_sha256):
    # earlier sealed-access rows for these weights
    path = sealed_log_path()
    if not os.path.exists(path):
        return []
    import csv
    with open(path, newline='') as fh:
        return [r for r in csv.DictReader(fh, delimiter='\t')
                if r['weights_sha256'] == weights_sha256]


def record_sealed_access(row):
    # header written when the file is new
    import csv
    path = sealed_log_path()
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=SEALED_LOG_COLS, delimiter='\t')
        if new:
            w.writeheader()
        w.writerow(row)
    return path


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
