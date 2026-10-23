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

- 2026-10-23 19:24 Smoke `kd9_10x70` (2 epochs of 10M shots, GPU otherwise idle apart from
  another user's llama-server at 0%): 165 s then 148 s per epoch, about 1,500 s per 100M-shot
  epoch alone. Trained, scored and selected end to end; the 8,000 MiB per-process cap held, so
  `GPU_MEM_MIB` stays 8000 as in the kd pilot.
- 2026-10-23 19:40 Epoch budget 160, not 240 (prereg exception). The d=7 depth sweep ran
  1.8-2.5x slower per epoch with several runs sharing the card, so three runs here project
  2,200-2,600 s per epoch: 240 epochs is 147-173 h, past the 120 h limit, and even the most
  optimistic 1,800 s per epoch lands exactly on it. 160 epochs projects 98-116 h. It also
  matches the budget of kd9_3x140, kd9_3x70 and kd9_1x140, so the comparisons share a schedule.
