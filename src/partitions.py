# Created: 2026-09-25
# Last modified: 2026-09-25
"""Read configs/partitions.yaml and configs/paths.yaml.

This is the only code that knows the partition ranges. Python scripts call the functions
below; shell launchers call the command-line form, which prints shell assignments:

    eval "$(python partitions.py --shell --d 7)"
    # -> TRAIN_START=0 TRAIN_STOP=15000000 VAL_START=15000000 VAL_N=200000
    #    EVAL_START=15200000 EVAL_N=1800000 SEALED_START=17000000 SEALED_STOP=19000000
    #    POOL=/home/.../data_d7_p0.004_r7_FORMAL.npz FORMAL_GEN_SEED=42 ...

Loading also checks that the four FORMAL ranges are disjoint and tile [0, n_total), so a
bad edit to the YAML fails immediately instead of producing overlapping blocks.
"""
import argparse
import os

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
