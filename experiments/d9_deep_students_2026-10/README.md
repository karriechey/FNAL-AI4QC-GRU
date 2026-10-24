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
- 2026-10-24 07:44 12 h in: kd9_10x70 and hl9_10x70 at epoch 29, kd9_8x70 at 34, so 1,490 s and
  1,270 s per epoch, as projected. /home 3.9 GB free, nothing saved yet (epochs start at 40).
- 2026-10-24 08:00 Added `resume_student.sh` and a `RESUME_OUT` branch in `run_student.sh`,
  after the 2026-10-22 reboot killed six depth-sweep runs mid-epoch. A resumed run continues in
  its own folder from the last complete epoch snapshot, which carries the Adam state, so the
  shot stream and the schedule are those of an uninterrupted run. Not deployed to grace1 while
  the three runs are live: bash reads a script as it executes, so overwriting the running copy
  can break it. On a crash the runs are already dead, so the deploy happens then, before
  `resume_student.sh` is used. A run killed before epoch 40 has no snapshot and restarts.
- 2026-10-24 17:40 Paused kd9_10x70, hl9_10x70 and kd9_8x70 at the next epoch boundary, by
  decision, not by a pre-registered rule: at epoch 52-61 the u70 stacks were near 5x MWPM on
  the validation block against 1.5x for kd9_3x140 at the same epoch. `pause_runs.sh` waits for
  the epoch's checkpoints to be written, stops the wrapper before the container so no
  half-trained checkpoint is scored, rebuilds the finished epochs as `resume_history.json` and
  writes PAUSED. `resume_student.sh` continues them exactly.
- 2026-10-24 17:45 Added kd9_taper4 (GRU 160-110-90-70, 293,261 parameters) as a dated prereg
  amendment, committed before launch. Shapes under the 303,521 cap spend one budget: 2 layers
  at width 170, 3 at 134, 4 at 114, 5 at 101, 6 at 92. The taper puts the width where the 80
  detector inputs arrive and narrows with depth.
- 2026-10-24 17:45 `pair_epoch.py` writes one status.log line every 10 epochs with this run's
  validation p_L and kd9_3x140's at the same epoch, from the kd pilot history.json.
- 2026-10-24 18:30 Smoke `kd9_taper4` (2 epochs of 10M shots, GPU free of our runs): 72 s then
  62 s per epoch, about 620 s per 100M-shot epoch alone, against 1,500 s for kd9_10x70. Four
  layers instead of ten is the difference; the parameter counts are within 2%. Trained, scored,
  selected and eval-scored end to end. 160 epochs projects about 28 h.
