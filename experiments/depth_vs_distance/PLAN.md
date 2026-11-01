<!-- Created: 2026-10-28 | Last modified: 2026-10-28 -->
# Depth vs distance

Fixed before any run of this experiment trained.

## Question

Does the minimum GRU depth needed to beat MWPM grow with code distance?

## Design

- d = 5, 7, 9; rounds = d; Z-basis memory; p = 0.004.
- Width scales with the input: width = round(3.5 x detectors per round), the same for every
  layer. Detectors per round d^2 - 1 = 24, 48, 80, so d5 u84, d7 u168, d9 u280.
- Depth L = 1-5 at each d.
- Parameter counts asserted with count_params() before each run (table below).

## Recipe (all runs)

Copied from the depth_d7_L*_u70 runs (experiments/depth_scaling_d7_2026-10/run_depth.sh):

- 100M fresh Stim shots every epoch, 8 sampling workers.
- Adam, peak LR from the scan below, warmup 3 epochs, cosine to 3e-5 over the epoch budget,
  global clipnorm 0.64, batch 10,000, GRU reset_after=False, Dense(1) logit, hard labels.
- TF32 on in training (kd pilot teacher check, d5/d7/d9 all "on"); every scored number with TF32 off.
- Validation block FORMAL [15.0M, 15.2M) logged every epoch, used only by the kill rule.
- Epoch budget: 80 at d5 and d7, 160 at d9.
- Fresh-shot seed bases (registered in configs/partitions.yaml): seed 0 runs 6e12 / 7e12 / 8e12
  (d5 / d7 / d9), shared by the LR scan and the L sweep at that d, so every depth sees the same
  shot stream; second-seed runs use init seed 1 and bases 9e12 / 10e12 / 11e12.

## Peak LR

LR scan at L = 1 and L = 5 for each d, peak 1e-3, 2e-3, 3e-3, 20 epochs of the full schedule
(80 or 160 epochs). Classification as lrscan_d7: diverged (kill rule fired or NaN), unstable
(training loss above 1.10 x its running minimum for 3 or more epochs), otherwise stable.
The L sweep at that d uses the highest LR stable at both L = 1 and L = 5. If none is, lrscan_d7's
conditional rule applies: 5e-4 at L = 1 and L = 5, used if stable at both; otherwise that d is
reported as having no stable LR and its L sweep is not run.

## Kill rule

Training loss NaN, or validation p_L above 1.5 x the run's best so far for 5 consecutive epochs
(streak_reason in experiments/mwpm_fixes_d9_2026-10/driver.py). A killed run is logged with its
epoch, is not scored, and the queue moves on.

## Scoring

- Every epoch is snapshotted (weights and Adam state) for resuming; epoch 20 and every epoch
  from 40 are kept and scored on the selection block FORMAL [0, 10M) with MWPM on the same shots.
- Selected checkpoint: lowest selection-block p_L from epoch 40, later epoch on a tie.
- Only that checkpoint is scored on the eval block [15.2M, 17M), GRU vs MWPM on identical shots,
  exact McNemar. The sealed block and the held-out pools are not read.
- Last-20-epoch slope of the selection-block ratio reported for every run; a run is flagged as
  still improving when the slope is negative with t < -2.

## L_min

L_min(d) = smallest L with eval ratio < 1 and McNemar p < 0.01 (seed 0). A second seed (init
seed 1, its own shot stream) is run at L_min - 1 and L_min (L_min only when L_min = 1). No
second seeds for a d where no L in 1-5 qualifies.

## Reuse

No existing run matches this recipe exactly, so nothing is reused. Close but not matched:

| run | d | shape | differs in |
|---|---|---|---|
| f2280 | 9 | 2 x u280 | checkpoint = min val p_L (not selection block); LR 1e-3 fixed in advance; 0.973x, p = 0.13 |
| f3280 | 9 | 3 x u280 | peak LR 5e-4 (outside the scan grid); val-p_L checkpoint; 0.652x |
| f3280s1 | 9 | 3 x u280 | as f3280, init seed 1, base 2e12; 0.633x |
| f3280e240 | 9 | 3 x u280 | LR 5e-4, 240 epochs; 0.623x |
| 9-1, 9-5, 9-7 | 9 | 1 x u280 | fixed 100M/200M training data, not fresh; val-p_L checkpoint |
| f5w_p004 | 5 | 1 x u280 | width 280 (not 84), LR 3e-3, no warmup |
| f7w_p004 | 7 | 2 x u280 | width 280 (not 168) |
| depth_d7_L*_u70 | 7 | L x u70 | width 70 (not 168) |

These already suggest the answer at d9: one u280 layer fails badly (about 2.5x on fixed data),
two layers reach parity, three beat MWPM clearly.

## Run table and cost

Estimated hours per run: s/epoch = T1 x (1 + 0.12 (L - 1)) with T1 = 350 / 550 / 800 s at
d5 / d7 / d9, from the fresh-shot runs (f5w 230, f7w 355, f2280 and f3280 610-640 s/epoch with
two or three runs on the card) scaled for 5-8 runs sharing the card, plus scoring (0.5 / 1 / 4 h).
Expect +-50%. Hours are wall-clock run-hours, the "runtime (h)" of the run tables; runs overlap,
so GPU-busy time is lower.

| run | d | L | width | parameters | peak LR | epochs | est. hours |
|---|---|---|---|---|---|---|---|
| lrscan_d5_L05_lr1e-3 | 5 | 5 | u84 | 197,905 | 1e-3 | 20 of 80 | 2.9 |
| lrscan_d5_L05_lr2e-3 | 5 | 5 | u84 | 197,905 | 2e-3 | 20 of 80 | 2.9 |
| lrscan_d5_L05_lr3e-3 | 5 | 5 | u84 | 197,905 | 3e-3 | 20 of 80 | 2.9 |
| lrscan_d5_L01_lr1e-3 | 5 | 1 | u84 | 27,553 | 1e-3 | 20 of 80 | 1.9 |
| lrscan_d5_L01_lr2e-3 | 5 | 1 | u84 | 27,553 | 2e-3 | 20 of 80 | 1.9 |
| lrscan_d5_L01_lr3e-3 | 5 | 1 | u84 | 27,553 | 3e-3 | 20 of 80 | 1.9 |
| dvd_d5_L01_u84 | 5 | 1 | u84 | 27,553 | from scan | 80 | 8.3 |
| dvd_d5_L02_u84 | 5 | 2 | u84 | 70,141 | from scan | 80 | 9.2 |
| dvd_d5_L03_u84 | 5 | 3 | u84 | 112,729 | from scan | 80 | 10.1 |
| dvd_d5_L04_u84 | 5 | 4 | u84 | 155,317 | from scan | 80 | 11.1 |
| dvd_d5_L05_u84 | 5 | 5 | u84 | 197,905 | from scan | 80 | 12.0 |
| second seeds at L_min - 1 and L_min | 5 | ? | u84 | | from scan | 80 | 20.3 (two runs, priced at L = 3) |
| lrscan_d7_L05_lr1e-3 | 7 | 5 | u168 | 788,929 | 1e-3 | 20 of 80 | 4.5 |
| lrscan_d7_L05_lr2e-3 | 7 | 5 | u168 | 788,929 | 2e-3 | 20 of 80 | 4.5 |
| lrscan_d7_L05_lr3e-3 | 7 | 5 | u168 | 788,929 | 3e-3 | 20 of 80 | 4.5 |
| lrscan_d7_L01_lr1e-3 | 7 | 1 | u168 | 109,537 | 1e-3 | 20 of 80 | 3.1 |
| lrscan_d7_L01_lr2e-3 | 7 | 1 | u168 | 109,537 | 2e-3 | 20 of 80 | 3.1 |
| lrscan_d7_L01_lr3e-3 | 7 | 1 | u168 | 109,537 | 3e-3 | 20 of 80 | 3.1 |
| dvd_d7_L01_u168 | 7 | 1 | u168 | 109,537 | from scan | 80 | 13.2 |
| dvd_d7_L02_u168 | 7 | 2 | u168 | 279,385 | from scan | 80 | 14.7 |
| dvd_d7_L03_u168 | 7 | 3 | u168 | 449,233 | from scan | 80 | 16.2 |
| dvd_d7_L04_u168 | 7 | 4 | u168 | 619,081 | from scan | 80 | 17.6 |
| dvd_d7_L05_u168 | 7 | 5 | u168 | 788,929 | from scan | 80 | 19.1 |
| second seeds at L_min - 1 and L_min | 7 | ? | u168 | | from scan | 80 | 32.3 (two runs, priced at L = 3) |
| lrscan_d9_L05_lr1e-3 | 9 | 5 | u280 | 2,188,481 | 1e-3 | 20 of 160 | 6.6 |
| lrscan_d9_L05_lr2e-3 | 9 | 5 | u280 | 2,188,481 | 2e-3 | 20 of 160 | 6.6 |
| lrscan_d9_L05_lr3e-3 | 9 | 5 | u280 | 2,188,481 | 3e-3 | 20 of 160 | 6.6 |
| lrscan_d9_L01_lr1e-3 | 9 | 1 | u280 | 303,521 | 1e-3 | 20 of 160 | 4.4 |
| lrscan_d9_L01_lr2e-3 | 9 | 1 | u280 | 303,521 | 2e-3 | 20 of 160 | 4.4 |
| lrscan_d9_L01_lr3e-3 | 9 | 1 | u280 | 303,521 | 3e-3 | 20 of 160 | 4.4 |
| dvd_d9_L01_u280 | 9 | 1 | u280 | 303,521 | from scan | 160 | 39.6 |
| dvd_d9_L02_u280 | 9 | 2 | u280 | 774,761 | from scan | 160 | 43.8 |
| dvd_d9_L03_u280 | 9 | 3 | u280 | 1,246,001 | from scan | 160 | 48.1 |
| dvd_d9_L04_u280 | 9 | 4 | u280 | 1,717,241 | from scan | 160 | 52.4 |
| dvd_d9_L05_u280 | 9 | 5 | u280 | 2,188,481 | from scan | 160 | 56.6 |
| second seeds at L_min - 1 and L_min | 9 | ? | u280 | | from scan | 160 | 96.2 (two runs, priced at L = 3) |

| d | est. run-hours |
|---|---|
| 5 | 85 |
| 7 | 136 |
| 9 | 370 |
| all | 591 |

With up to 8 runs at once and the d9 chain (scan, then L = 5 at about 57 h, then a second seed)
as the longest path, the queue should take about 5-7 days of wall clock.

## Execution

- queue.sh runs everything unattended in launch order: d5 scan, d5 L = 1-5, d7 scan, d7 L = 1-5,
  d9 scan, d9 L = 1-5, second seeds per d. A launch waits only on what it needs.
- Resume-safe: rerunning queue.sh skips finished runs and resumes unfinished ones from their last
  epoch snapshot.
- After each run: GRU_depth_vs_distance_runs.csv (the GRU_depth_scaling_runs_2026-10-23 columns
  plus width/detectors and seed) and STATUS.md are regenerated.
- At the end: lmin/d*.json, figures/fig_depth_vs_distance.py -> fig_depth_vs_distance.png/.pdf.

Deviations from the request, forced by grace1:

- tmux is not installed on grace1. queue.sh runs under setsid nohup, which also survives the end
  of the session (same as the depth-scaling launchers).
- /home is full (2.8 GB free of 1.4 TB, shared). Epoch snapshots go to /tmp/kchey_dvd (48 GB
  free); each run folder links to its snapshot folder. The queue touches them daily against the
  10-day /tmp cleanup. Selected weights, logs and scores stay in the run folders under /home.
- grace1 cannot write to Google Sheets. The CSV on grace1 is the record; rows go to the sheet tab
  GRU_depth_vs_distance whenever it is synced from the Mac.

## Commands (grace1, from this folder)

    setsid nohup bash queue.sh > queue/queue.out 2>&1 < /dev/null &
    cat STATUS.md; tail status.log
    touch STOP            # no further launches; running runs finish
