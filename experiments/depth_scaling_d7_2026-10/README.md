# GRU depth scaling, d = 7, p = 0.004

Created: 2026-10-21
Last modified: 2026-10-21

L x GRU u70 for L = 1-10, plus single layers matched in parameters to L = 4 and L = 8, after an
LR stability scan at L = 1 and 10. Design, run tables and decision rules:
`prereg/depth_scaling_d7_2026-10.md`. Results: `GRU_depth_scaling_runs_<date>.csv` at the repo
root, figures from `figures/depth_scaling.py`.

| file | what |
|---|---|
| `run_depth.sh` | one run by name: parameter assertion, resource wait, training, kill rule, scoring |
| `launch_lr_scan.sh` | Part A: eight scan runs, the sweep LR rule, the conditional 5e-4 run |
| `launch_depth_sweep.sh` | Part B: waits for the sweep LR and `depth_sweep/GO`, queues the depth and matched runs, runs `report.py` |
| `classify_lr_scan.py` | kill rule (`--kill-check`), stable / unstable / diverged, `lr_scan/sweep_lr.json` |
| `report.py` | results CSV and `depth_sweep/report.md` (L*, depth vs parameters, degradation) |
| `status.log` | one line per event; a stop reason goes on top |
| `lr_scan/`, `depth_sweep/` | one folder per run, `<run>_<UTC stamp>` |

Reuses `experiments/selection_2026-10-14/score_selection.py`,
`experiments/kd_pilot_p004_2026-10/select_checkpoint.py`, `streak_reason` from
`experiments/mwpm_fixes_d9_2026-10/driver.py` and `src/pair_two_decoders.py`.

## Commands (grace1, from this folder)

    SMOKE=1 bash run_depth.sh depth_d7_L10_u70
    setsid nohup bash launch_lr_scan.sh > lr_scan/launch.out 2>&1 < /dev/null &
    setsid nohup bash launch_depth_sweep.sh > depth_sweep/launch.out 2>&1 < /dev/null &
    touch depth_sweep/GO        # after the Part A amendment is committed
    ~/tfenv/bin/python report.py

## Decisions during the run

Times UTC.

- 2026-10-21 06:03 Smoke `depth_d7_L10_u70` (2 epochs of 10M shots, GPU otherwise idle): epoch 2
  103 s (about 1,030 s per 100M-shot epoch alone), GPU utilization about 30%, training peak
  4.7 GB GPU and 4.6 GB host; selection scoring peak 20 GB host, 8.6 GB GPU. Smoke
  `lrscan_d7_L10_lr5e-3` checked the stop-after path (stopped after epoch 1 of 2, COMPLETE).
  Set `GPU_MEM_MIB=6000`, `NEED_HOST_GB=30`, `MAX_CONCURRENT=12`: all of Part A, and later all
  of Part B, run at once (12 x 6 GB fits the 98 GB card; 72 cores for 8 sampling workers per run).
- Launcher: with `SKIP_TAIL_EVAL=1` the final summary `grep` found no eval CSV and `pipefail`
  made the launcher exit 2 after writing COMPLETE; `|| true` added.
- The GPU check counts only other users' processes, so these runs share the card with each
  other; the 5-minute idle check applies when another user is on the GPU and none of these
  runs is up.
- /home had 6.1 GB free at launch (shared 1.4 TB volume). Expected writes about 2 GB
  (epochs 20-80 of 12 runs, 0.7-3.5 MB per epoch file); each start needs 3 GB free.
- Part B waits up to 3 h for `depth_sweep/GO` after the scan, so the amendment can be
  committed first; without it Part B starts and the amendment follows.
- Commits are local on branch `depth-scaling`, not pushed. Code on grace1:
  `~/FNAL-AI4QC-GRU-depth`, a `git archive` of the commit in `CODE_COMMIT.txt`.
- 2026-10-21 07:20 Part A, all 8 runs at once: L=10 1,720-1,775 s per epoch, L=1 about 450 s,
  GPU utilization 100%, 26 GB GPU and 134 GB host in use. Expected end of Part A about 16:00.
- 2026-10-21 08:37 `lrscan_d7_L01_lr5e-3` killed at epoch 17 by the kill rule: best val p_L 0.0211
  (epoch 7), epochs 13-17 at 0.032-0.040 with training loss up from 0.058 to 0.088. Classified
  diverged. The sweep LR rule uses only L=10, so the scan continues unchanged.
- 2026-10-21 08:55 L=1 at 1e-3, 2e-3 and 3e-3 finished 20 epochs.
- 2026-10-21 10:21 `lrscan_d7_L10_lr5e-3` killed at epoch 9: best val p_L 0.0160 (epoch 2),
  epochs 5-9 at 0.027-0.088 with training loss up from 0.044 to 0.180. Classified diverged.
- 2026-10-21 13:48-19:00 no access to grace1 (Kerberos ticket expired). Part A ended 13:54 with
  sweep LR 1e-3; Part B started on its own at 16:55 after the 3-hour GO wait; the amendment was
  committed at 19:10, after Part B started.
- 2026-10-21 19:05 Part B, 11 runs at once (L=2 reused): epoch 1 took 731 s (L=1) to 2,565 s
  (L=10), u172 876 s, u255 921 s; GPU 29 GB at 100%, host 146 GB, /home 6.0 GB free. L=10 at 80
  epochs: about 55 h at this rate, less as the shorter runs finish.
- 2026-10-22 03:40-13:00 no access to grace1 from the workstation (network); runs unaffected.
  Scoring after training takes about 12 min per run (42 selection checkpoints, then eval).
- 2026-10-22 21:19 grace1 rebooted (uptime since 21:19Z, no notice in the run logs). L=1, L=3,
  L=4, u172 and u255 were already done. The other six died mid-epoch: L=5 after 68 epochs,
  L=6 after 62, L=7 after 53, L=8 after 50, L=9 after 46, L=10 after 43.
- 2026-10-23 02:54 resumed those six in their own folders from `epochs/epoch<N>.weights.h5`,
  which carries the Adam slots (`optimizer/vars` in the h5). `--initial-epoch N` restores the
  update count (L=10: 430,000 = 10,000 x 43, asserted by train_student.py) and the fresh-shot
  stream continues at epoch N, so the shots match an uninterrupted run. New files:
  `launch_resume_after_reboot.sh`, `history_from_log.py`; `run_depth.sh` gains `RESUME_OUT`
  and `FOREIGN_MAX_MIB`.
  - `.lastepoch.weights.h5` was a partial write for L=9, as was its `epoch047.weights.h5`
    (h5 "truncated file"), so the resume point is the last complete epoch line in run.log.
  - The launcher truncates run.log, so the earlier log is kept as `run_part1.log` and
    `history_from_log.py` rebuilds the finished epochs as `--resume-history`, which keeps
    history.json at 80 epochs for the COMPLETE check and the last-20-epoch slope.
  - The kill rule now sees only the resumed epochs, so its "best so far" restarts at epoch N+1.
- 2026-10-23 02:54 another user's `llama-server` holds 57 GB on the card at 0% utilization,
  above the 10 GB limit in the resource check. Raised to `FOREIGN_MAX_MIB=70000` for the
  resumed runs: 40 GB free covers six runs at about 4.7 GB each, and the 5-minute idle check
  (mean 0%) still gates each start.
