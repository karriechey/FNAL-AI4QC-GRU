# Created: 2026-09-25
# Last modified: 2026-09-25
"""Old-code half of the A5 equivalence tests. Runs in its own Python process.

The old repo (FNAL-QCDecoding-FPGA @ d5ee86f) and this repo have modules with the same
names (StudentModels, circuit_generators, generate_pools, ...). Importing both into one
process would make one silently shadow the other, so a5_equivalence.py runs this script
as a subprocess with the old repo first on sys.path, and compares its output files with
what the new code produces.

This script only reads the old repo; it writes nothing there.

    python old_side.py --old-repo ~/FNAL-QCDecoding-FPGA --task circuits --out old.json
    python old_side.py --old-repo ... --task scatter --slice S.npz --d D --out old.npz
    python old_side.py --old-repo ... --task logits  --slice S.npz --d D --weights W --out old.npz

--slice is a small .npz holding `det` (the eval-block rows under test), written by
a5_equivalence.py, so this process never has to load a whole 19M-shot pool.
"""
import argparse
import hashlib
import json
import os
import sys


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old-repo', required=True)
    ap.add_argument('--task', choices=['circuits', 'scatter', 'logits'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--p', type=float, default=0.004)
    ap.add_argument('--d', type=int, default=5)
    ap.add_argument('--slice', default=None, help='.npz with key det')
    ap.add_argument('--weights', default=None)
    ap.add_argument('--units', type=int, default=140)
    a = ap.parse_args()

    old = os.path.expanduser(a.old_repo)
    sys.path.insert(0, old)
    os.chdir(old)            # old modules resolve relative paths from their own folder

    if a.task == 'circuits':
        # Every place the old repo built the training/eval circuit that the GRU pipeline
        # uses: eval_on_tail.build_circuit (MWPM, pool generation, eval) and
        # generate_pools.build_circuit (sampling helper). StudentModels builds the same
        # circuit inline for the detector layout; that path is covered by the scatter test.
        import eval_on_tail
        import generate_pools
        out = {}
        for d in (5, 7, 9):
            for name, circ in (('eval_on_tail', eval_on_tail.build_circuit(d, a.p, d)),
                               ('generate_pools', generate_pools.build_circuit(d, a.p))):
                out[f'{name}_d{d}'] = dict(
                    circuit_sha=sha(str(circ)),
                    dem_sha=sha(str(circ.detector_error_model(decompose_errors=True))),
                    num_detectors=circ.num_detectors)
        with open(a.out, 'w') as fh:
            json.dump(out, fh, indent=1)
        return

    import numpy as np
    det = np.load(a.slice)['det']

    if a.task == 'scatter':
        from StudentModels import to_sequence, detector_sequence_layout
        seq = to_sequence(det, a.d, a.d, a.p)
        _, _, index_map = detector_sequence_layout(a.d, a.d, a.p)
        np.savez(a.out, seq=seq, index_map=index_map)
        return

    if a.task == 'logits':
        from StudentModels import to_sequence, build_gru_student
        model = build_gru_student(d=a.d, rounds=a.d, units=a.units, p=a.p)
        model.load_weights(os.path.expanduser(a.weights))
        x = to_sequence(det, a.d, a.d, a.p)
        logits = model.predict(x, batch_size=10000, verbose=0).reshape(-1)
        np.savez(a.out, logits=logits, n_params=model.count_params())
        return


if __name__ == '__main__':
    main()
