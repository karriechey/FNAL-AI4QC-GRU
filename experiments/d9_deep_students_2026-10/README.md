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

- 2026-10-25 14:32 kd9_taper4 paused at the end of epoch 120 of 160, by decision, not by a rule
  in the prereg. At epoch 110 it was at 2.87x MWPM on the validation block against 1.10x for
  kd9_3x140 at the same epoch, gaining about 0.1x per 10 epochs, so the last 40 epochs cannot
  change the ordering. Weights, Adam state and history are kept; `resume_student.sh kd9_taper4`
  continues it exactly. It is resumed after kd9_taper3_t100 finishes and scored then, so its
  160-epoch number still enters the table.
- 2026-10-25 kd9_taper3_t100 (GRU 160-120-100, 283,241 parameters) added as a dated prereg
  amendment and launched alone on the GPU. The instruction that added it said both to launch
  after kd9_taper4 finished and scored, and to pause kd9_taper4 and launch now; the later and
  more specific of the two was followed, so taper4's remaining epochs move behind taper3.
- 2026-10-25 14:46 Smoke `kd9_taper3_t100` (2 epochs of 10M shots, GPU free of our runs): 57 s
  then 48 s per epoch, about 480 s per 100M-shot epoch alone, against 620 s for kd9_taper4 and
  1,500 s for kd9_10x70. Parameter assertion passed at 283,241. Trained, scored, selected and
  eval-scored end to end. 160 epochs projects about 21 h, well inside the 120 h limit, so the
  budget stays 160.
- 2026-10-25 20:00 `pair_epoch.py` corrected twice over. It read `val_ler` from the Keras epoch
  line, which prints to four decimals, so two runs 12 validation errors apart both showed 0.0091
  at epoch 35 and their ratios to kd9_3x140 came out identical to three decimals. It now reads
  `val_errors`, an integer count over the 200,000-shot validation block, and divides, which is
  exact. Second, `pause_runs.sh` copies run.log to run_part1.log, so between a pause and a resume
  the two files are identical and the concatenation counted kd9_taper4's 120 epochs twice. Epochs
  up to 120 still indexed correctly, so no pairing line written so far was wrong, but a resumed
  run would have mis-indexed past 120. Identical files are now counted once, verified: 120 epochs
  parsed for kd9_taper4, not 240.
- 2026-10-25 Train and validation loss track at matched epochs for both tapers (ratios to
  kd9_3x140 at epoch 35 agree to about 1%), as 100M fresh shots per epoch imply, so the deficit
  is on the training objective rather than generalization.
- 2026-10-25 20:10 `pair_epoch.py` rewritten to key on the epoch number train_student.py prints
  rather than on file order, keeping one entry per epoch and failing if the epochs are not exactly
  1...N or if an epoch appears twice with different values. Checked against every run folder: the
  four paused runs (kd9_10x70 54 epochs, hl9_10x70 55, kd9_8x70 64, kd9_taper4 120) and
  kd9_taper3_t100 (39) all parse contiguous with no duplicates, so the duplicate-log fault is
  covered for all of them, not only kd9_taper4. p_L now comes from the integer `val_errors` count
  over the 200,000-shot block; the kd pilot reference reads `val_errors` from history.json and
  falls back to val_ler x 200,000 for an older history without the counts.
- 2026-10-25 The selected-checkpoint and eval-block numbers were never exposed to the log fault.
  `score_selection.py` loads each `epochs/epochNNN.weights.h5` and decodes the selection block
  itself, `select_checkpoint.py` reads only that scoring CSV and asserts the weights path it picked
  matches the epoch, and the eval block is scored from the selected weights file. Neither reads
  run.log, val_ler or history.json. The paired status.log lines are monitoring only.
- 2026-10-25 20:02 kd9_taper3_t100 paused at the end of epoch 40, by decision, not by a rule in
  the prereg: it tracked kd9_taper4 to within 1% at every matched epoch while kd9_3x140 was at
  1.82x. Resumable exactly; resumed and scored after kd9_3x134.
- 2026-10-25 kd9_3x134 (3 x GRU 134, 302,841 parameters, 240 epochs) added as a dated prereg
  amendment. It is the widest uniform three-layer stack inside the 303,521 budget, so it separates
  uniform width from taper at depth 3. Its 240-epoch budget is set in the run table in
  `run_student.sh`, not in epochs.txt, which stays 160 for the other runs.
- 2026-10-25 20:18 Smoke `kd9_3x134` (2 epochs of 10M shots, GPU free of our runs): 56 s then
  47 s per epoch, about 470 s per 100M-shot epoch alone. Parameter assertion passed at 302,841.
  Trained, scored, selected and eval-scored end to end. 240 epochs projects about 31 h, inside
  the 120 h limit, so the budget stays 240. Epoch weights from epoch 40 are about 0.72 GB against
  4 GB free on /home, checked again every 12 h.
- 2026-10-25 20:24 kd9_3x134 stopped before training: "240 epochs from base 5000000000000 exceed
  its registered max_epochs 160", the seed-range guard in `src/fresh_shots.py` against two runs
  drawing overlapping fresh-shot seeds. The 240 in the amendment came from a wrong reading of
  kd9_3x140, which trained 160 epochs; 240 is the teacher's budget, on seed base 1 x 10^12. Fixed
  by running kd9_3x134 for 160 epochs, which also matches kd9_3x140's schedule and shot stream
  exactly, rather than by extending the reservation. Prereg corrected before the relaunch. The
  stopped run folder is kept with no weights.
- 2026-10-26 01:10 kd9_3x134 stopped at epoch 33 of 160 and restarted at 240 epochs, the budget of
  the amendment that added it. Running it at 160 was my choice after the seed-range guard refused
  240, and it was not mine to make. Base 5 x 10^12 is now reserved for 240 epochs in
  `configs/partitions.yaml`, which is safe because the bases are 1 x 10^12 apart and 240 epochs
  span 2.4 x 10^8 seeds. The 160-epoch folder is kept unscored with its epoch weights removed.
  Pairing lines against kd9_3x140 are a progress trace from here, not a schedule-matched
  comparison, since kd9_3x140 ran 160 epochs.
- 2026-10-27 09:23 kd9_3x134 done at 240 epochs. Selected epoch 237 of 240 (selection block
  1.067x), eval block p_L 0.002391 against MWPM 0.002267, ratio 1.0546x at 302,841 parameters.
  kd9_3x140 is 1.0461x at 329,001. Paired McNemar on the eval block: against MWPM net -223 shots,
  exact p 0.0020; against the teacher f3280e240 net -1,761, p 1.8e-203; against kd9_3x140 net -35
  of 1.8M shots, p 0.58, so the two are indistinguishable and the d=9 student fits inside the
  303,521 budget. Convergence test: last-20 slope -5.7e-4 per epoch, t = -4.74, not converged, so
  it was still improving at the end of the budget. Tests written to `mcnemar/`.
- 2026-10-27 12:04 grace1 rebooted again (boot 07:04 local) and killed kd9_taper3_t100 in epoch 60
  of 160, nine epochs into the stretch that resumed at 09:24. No OOM in dmesg. kd9_3x134 had
  finished and been scored at 09:23, so nothing of it was lost. Resumed from the epoch 59 snapshot,
  optimizer update count 590,000 = 10,000 x 59 as asserted.
- 2026-10-27 15:14 Two faults in the resume path, both from this being a second resume of one run.
  `resume_student.sh` took the last epoch from a line count of run.log, which after a resume starts
  again at the resume epoch, so it would have anchored on epoch 19 rather than 59; it now takes the
  largest epoch number train_student.py printed across every part log, counting an epoch complete
  only when its grad-norm line is followed by the Keras timing line, and rotates run.log into the
  next `run_partN.log` so an earlier stretch is never overwritten. `history_from_log.py` accepts
  several --log parts for the same reason. The same line count drove the wrapper's status and
  pairing epochs, which is why the first stretch logged "epoch 18" and a mislabeled "paired epoch
  10"; `run_student.sh` now reads the printed epoch number. The stretch was restarted from epoch 59
  under the fixed scripts, costing one incomplete epoch.
