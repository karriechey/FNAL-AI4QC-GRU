# Deep narrow d=9 students, p = 0.004

Created: 2026-10-23
Last modified: 2026-10-23

10 x GRU 70 and 8 x GRU 70 at d=9, distilled from f3280e240, plus the hard-label 10 x GRU 70.
Design, runs and rules: `prereg/d9_deep_students_2026-10.md`. Results:
`GRU_d9_deep_students_runs_<date>.csv` at the repo root.

| file | what |
|---|---|
| `run_student.sh` | one run by name: parameter assertion, resource wait, training, kill rule, scoring, pruning |
| `launch.sh` | queues the three runs, staggered |
| `epochs.txt` | the epoch budget the smoke test settled on; read by `run_student.sh` |
| `status.log` | one line per event; a stop reason goes on top |
| `runs/`, `smoke/` | one folder per run, `<run>_<UTC stamp>` |

Reuses `experiments/kd_pilot_p004_2026-10/` (`teachers.sh`, `select_checkpoint.py`,
`report.py`, the d9 teacher check and the teacher copy),
`experiments/selection_2026-10-14/score_selection.py` and the kill rule from
`experiments/depth_scaling_d7_2026-10/classify_lr_scan.py`.

## Commands (grace1, from this folder)

    SMOKE=1 bash run_student.sh kd9_10x70
    setsid nohup bash launch.sh > runs/launch.out 2>&1 < /dev/null &
    ~/tfenv/bin/python ../kd_pilot_p004_2026-10/report.py --d 9 --base runs

## Decisions during the run

Times UTC.
