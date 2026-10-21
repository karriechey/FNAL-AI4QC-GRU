# Created: 2026-09-26
# Last modified: 2026-09-26

"""MWPM logical error rate vs physical error rate, for the threshold.

Same circuit and decoder as every paired MWPM baseline (circuit_generators.build_circuit).
Fresh sinter samples; no pool is read. Stops per (d, p) at 1,000 errors or 10M shots, and
can resume from the CSV.

python threshold_sweep.py --out <dir>/stats.csv --workers 64
python threshold_sweep.py --out <dir>/timing.csv --max-shots 100000 --max-errors 1000000000  # timing run
"""
import argparse
import os

import sinter

from circuit_generators import build_circuit

P_GRID = (0.003, 0.004, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.008, 0.009)
DISTANCES = (5, 7, 9)


def make_tasks(distances, p_grid):
    tasks = []
    for d in distances:
        for p in p_grid:
            circ = build_circuit(d, p, d)
            tasks.append(sinter.Task(
                circuit=circ,
                detector_error_model=circ.detector_error_model(decompose_errors=True),
                json_metadata=dict(d=d, p=p, r=d, basis='Z',
                                   circuit='surface_code:rotated_memory_z')))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='CSV of sinter stats (resumable)')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--max-shots', type=int, default=10_000_000)
    ap.add_argument('--max-errors', type=int, default=1_000)
    ap.add_argument('--distances', type=int, nargs='*', default=list(DISTANCES))
    ap.add_argument('--p', type=float, nargs='*', default=list(P_GRID))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tasks = make_tasks(a.distances, a.p)
    print(f"[sweep] {len(tasks)} tasks  d={a.distances}  p={a.p}  workers={a.workers}  "
          f"stop at {a.max_errors:,} errors or {a.max_shots:,} shots", flush=True)
    sinter.collect(num_workers=a.workers, tasks=tasks, decoders=['pymatching'],
                   max_shots=a.max_shots, max_errors=a.max_errors,
                   save_resume_filepath=a.out, print_progress=True)
    print(f"[sweep] done -> {a.out}", flush=True)


if __name__ == '__main__':
    main()
