# Created: 2026-09-26
# Last modified: 2026-09-26
"""C1: list every GRU best checkpoint under the results root, read from the weight files.

    python tests/c1_list_checkpoints.py [results_root] > c1_checkpoints.txt

For each run folder: GRU units and input width (from the kernel shape), reset_after (bias
shape (3u,) means False, (2, 3u) means True), parameter count, and which seeds exist.
The time axis is not stored in the weights; it follows from d (r + 1 = d + 1 timesteps).
"""
import collections
import glob
import os
import sys

import h5py
import numpy as np

root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/rcnn_threshold/results')
D_BY_WIDTH = {24: 5, 48: 7, 80: 9}
rows = collections.OrderedDict()
for f in sorted(glob.glob(os.path.join(root, '**', '*gru*.best.weights.h5'), recursive=True)):
    shapes = {}
    with h5py.File(f, 'r') as h:
        h.visititems(lambda n, o: shapes.__setitem__(n, o.shape) if hasattr(o, 'shape') else None)
    # Two layouts: the .weights.h5 format (layers/<name>/cell/vars/0,1,2 = kernel,
    # recurrent kernel, bias; also stores optimizer and metric state) and the legacy h5
    # format (.../gru_cell/kernel:0, bias:0).
    layer = {n: s for n, s in shapes.items() if n.startswith('layers/') or 'model_weights' in n
             or ':0' in n}
    kern = [s for n, s in layer.items()
            if n.endswith('gru_cell/kernel:0') or (('gru' in n) and n.endswith('cell/vars/0'))]
    bias = [s for n, s in layer.items()
            if n.endswith('gru_cell/bias:0') or (('gru' in n) and n.endswith('cell/vars/2'))]
    if not kern:
        continue
    n_in, three_u = kern[0]
    units = three_u // 3
    reset_after = len(bias[0]) == 2
    params = int(sum(np.prod(s) for s in layer.values() if len(s)))
    run = os.path.relpath(f, root).split(os.sep)[0]
    seed = os.path.basename(f).split('_seed')[1].split('_')[0] if '_seed' in f else '?'
    key = (run, units, n_in, reset_after, params, str(bias[0]))
    rows.setdefault(key, []).append(seed)

print(f"{'run folder':56s} {'units':>5s} {'input':>8s} {'reset_after':>11s} {'params':>8s} "
      f"{'bias':>9s} seeds")
for (run, u, n_in, ra, p, b), seeds in rows.items():
    d = D_BY_WIDTH.get(n_in)
    shape = f'({d + 1},{n_in})' if d else f'(?,{n_in})'
    print(f"{run:56s} {u:5d} {shape:>8s} {str(ra):>11s} {p:8,d} {b:>9s} "
          f"{','.join(sorted(set(seeds)))}")
