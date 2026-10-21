<!-- Created: 2026-10-21 | Last modified: 2026-10-21 -->
# Pre-registration: GRU depth scaling at d=7 (October 2026)

Committed before any run of this experiment trained. Only a 2-epoch timing smoke
(10M shots per epoch, `smoke/`, not part of any result) ran before this file. Code and run
folders: `experiments/depth_scaling_d7_2026-10/`.

## Question

At fixed width (u70), how does p_L / MWPM change as GRU layers are stacked from 1 to 10 at
d=7, and is the gain from depth or from parameter count? First, find a peak LR that is stable
for the deepest stack.

## Fixed for every run

- d=7, r=7, p=0.004. GRU reset_after=False, stacked layers (`--units 70 70 ...`,
  `StudentModels.build_gru_student`), then a Dense(1) logit.
- Hard labels (alpha 1, no teacher). Initialisation seed 0. Batch 10,000. Adam, global
  clipnorm 0.64. Warmup 3 epochs, then cosine to 3e-5, schedule laid over 80 epochs.
- 100M fresh Stim shots per epoch, fresh-shot seed base 4 x 10^12, the d7 base of the kd
  pilot (`experiments/kd_pilot_p004_2026-10/`). Every run here, and the pilot's hl7_2x70, sees
  the same shot stream.
- TF32 in training as decided by the kd pilot teacher check at d7 (`teacher_check/d7/
  tf32_decision.txt` on grace1: on). Every scored number is computed with TF32 off.
- Validation block FORMAL [15.0M, 15.2M) of the d7 p=0.004 FORMAL pool (generation seed 42):
  logged every epoch, used only by the kill rule.
- Parameter budget: none. L=10 at u70 has 291,551 parameters, above the d7 u280 deployment
  budget. This is a capacity study, not a deployment candidate, so `MAX_PARAMS` is not set for
  these runs.
- `gh200_run_student_scaling.sh` gains two pass-throughs, both off by default:
  `STOP_AFTER_STEPS` (train_student.py `--stop-after-steps`) and `SKIP_TAIL_EVAL` (no
  evaluation-block scoring of the validation-best checkpoint after training). Every run here
  sets `SKIP_TAIL_EVAL=1`, so the eval block is read only for the selected checkpoint.

Parameter counts, asserted with `count_params()` by `run_depth.sh` before every run; a
mismatch stops the experiment.

| model | params |
|---|---|
| L=1 u70 | 25,061 |
| L=2 u70 | 54,671 |
| L u70 | 25,061 + 29,610 x (L-1) |
| L=4 u70 | 113,891 |
| L=8 u70 | 232,331 |
| L=10 u70 | 291,551 |
| 1 x u172 | 114,209 (matches L=4) |
| 1 x u255 | 232,816 (matches L=8) |

## Part A: LR stability scan (runs first)

| runs | L | width | peak LR | eta x N_det (N_det = 336) | length |
|---|---|---|---|---|---|
| 8 | 1, 10 | u70 | 1e-3, 2e-3, 3e-3, 5e-3 | 0.34, 0.67, 1.0, 1.7 | 20 epochs |
| 1, conditional | 10 | u70 | 5e-4 | 0.17 | 20 epochs, only if L=10 at 1e-3 is not stable |

Run names `lrscan_d7_L01_lr1e-3` ... `lrscan_d7_L10_lr5e-3`, `lrscan_d7_L10_lr5e-4`.

- The schedule is the 80-epoch one (warmup 3, cosine to 3e-5 at epoch 80). The run stops after
  epoch 20 (`STOP_AFTER_STEPS=200000`), so epochs 1-20 see the same LR as in a full run.
- Gradient norms logged every update (implied by the clipnorm).
- Kill rule, as `prereg/mwpm_fixes_d9_2026-10.md`: NaN training loss, or validation p_L above
  1.5 x the run's best validation p_L so far for 5 consecutive epochs. Checked every minute
  from the run log (`classify_lr_scan.py --kill-check`, reusing `streak_reason` from
  `experiments/mwpm_fixes_d9_2026-10/driver.py`).
- Classification (`classify_lr_scan.py`), per run, from the per-epoch training loss
  (history.json, or the run log for a killed run):
  - diverged: killed by the kill rule, or NaN loss
  - unstable: training loss above 1.10 x its running minimum (over earlier epochs) for 3 or
    more consecutive epochs, without meeting the kill rule
  - stable: neither
- Sweep LR rule: 1e-3 if L=10 at 1e-3 is stable; otherwise 5e-4 if stable; if neither is
  stable, the depth sweep is not run and the experiment stops with a report.
- LR-scan runs are not scored on the selection block or the eval block.

When the scan finishes, the classification table is added below as a dated amendment and
committed, then Part B starts.

## Part B: depth sweep and parameter-matched controls

| runs | model | names | LR | length |
|---|---|---|---|---|
| 9 | L = 1, 3, 4, 5, 6, 7, 8, 9, 10 at u70 | `depth_d7_L01_u70` ... `depth_d7_L10_u70` | sweep LR | 80 epochs |
| 1, conditional | L = 2 at u70 | `depth_d7_L02_u70` | sweep LR | 80 epochs, only if the sweep LR is not 1e-3 |
| 2 | 1 x u172, 1 x u255 | `match_d7_L04_u172`, `match_d7_L08_u255` | sweep LR | 80 epochs |

If the sweep LR is 1e-3, L=2 is the kd pilot's `hl7_2x70` (same recipe, seed base and shots).
Its epochs were saved from 40, so it has no epoch-20 selection score; its validation-best
checkpoint was also scored on the eval block by the pilot launcher. Neither changes its
selected checkpoint or the selected checkpoint's eval score, which are used here.

Checkpoints and scoring (as the kd pilot, except the save start):

- Every epoch from 20 on is saved. Epochs 20 and 40-80 are scored on the selection block
  FORMAL [0, 10M) with MWPM on the same shots (`experiments/selection_2026-10-14/
  score_selection.py`). Epochs 20, 40, 60, 80 are the learning-curve points.
- Selected checkpoint: lowest selection-block p_L over epochs 40-80, later epoch on a tie
  (`experiments/kd_pilot_p004_2026-10/select_checkpoint.py`).
- Only the selected checkpoint is scored on the eval block FORMAL [15.2M, 17M)
  (`eval_student_on_tail.py --no-tf32`, per-shot dump). The sealed block [17M, 19M) and the
  held-out blocks are not read.
- Convergence: least-squares slope of the selection-block ratio over epochs 61-80; not
  converged if the slope is negative with t < -2.
- Paired McNemar on the eval block (`src/pair_two_decoders.py`, shots aligned by index):
  each run vs MWPM, each L vs L-1, `match_d7_L04_u172` vs L=4, `match_d7_L08_u255` vs L=8.
  b, c, n_disc and exact p are reported.
- The Part A kill rule also applies to Part B runs. A killed run is reported as killed at its
  epoch and is not scored.

## Decision rules (Part B)

Single seed per depth. Same-recipe seed spread on the eval block: f3280 seed 0 vs 1, 0.652x vs
0.633x (2.9%); d5 u140 Exp 16, three seeds at 10M 1.112-1.160x (4.3%), at 20M 0.941-0.957x
(1.7%). Gaps under 3% are therefore unresolved.

- L*: the smallest L in 1-9 where L+1 improves the eval-block p_L / MWPM by less than 3%
  (relative to L) or the L+1 vs L McNemar exact p >= 0.01. If no L qualifies, L* = 10 and depth
  had not saturated by L=10.
- Depth vs parameters: a stack beats its matched single layer when its eval-block ratio is
  lower by at least 3% and the McNemar exact p < 0.01. The reverse with the same thresholds
  means the single layer wins; otherwise unresolved.
- Degradation at large L: an L degrades when its ratio is at least 3% above L-1 with McNemar
  p < 0.01. For each such L, final-epoch training loss vs L-1: more than 1% higher is
  optimization failure; within 1% or lower is unused capacity. No residual or LayerNorm
  variants in this experiment.

## Launching and resources

- Smoke first: `SMOKE=1 bash run_depth.sh depth_d7_L10_u70` (2 epochs of 10M shots). Seconds
  per epoch and peak GPU and host memory set `GPU_MEM_MIB`, `NEED_HOST_GB` and the number of
  concurrent runs.
- Before every start, `run_depth.sh` waits (re-checking every 5 minutes) until: free space on
  /home at least 3 GB; host MemAvailable at least `NEED_HOST_GB`; fewer than `MAX_CONCURRENT`
  of these runs up; free GPU memory at least `GPU_MEM_MIB` + 2 GB; other users' GPU processes
  under 10 GB, and, when none of these runs is up, GPU utilization near 0% over 5 minutes
  (mean < 5%, max < 20%), as `experiments/mwpm_fixes_d9_2026-10/driver.py`.
- Every job is detached (setsid nohup). One status line per event in `status.log`.
- The experiment stops only if neither 1e-3 nor 5e-4 is stable at L=10, a parameter
  assertion fails, or something breaks the protocol; the reason goes at the top of
  `status.log`.

## Outputs

- `GRU_depth_scaling_runs_<date>.csv` at the repo root (`report.py`).
- `figures/depth_scaling.py`: eval-block p_L / MWPM vs L, and LR stability vs depth.

## Amendments

### 2026-10-21: Part A result

Part A finished 2026-10-21 13:54 UTC. Classification by the rules above (all 20-epoch runs on
the 80-epoch schedule; grad norm and clip fraction are per-epoch medians and fractions of steps
clipped at 0.64):

| run | L | peak LR | eta x N_det | status | epochs | final train loss | longest rise (epochs) | max loss / running min | max epoch grad norm median | max epoch clip fraction |
|---|---|---|---|---|---|---|---|---|---|---|
| lrscan_d7_L01_lr1e-3 | 1 | 1e-3 | 0.34 | stable | 20 | 0.04995 | 0 | 0.99 | 0.1463 | 0.0013 |
| lrscan_d7_L01_lr2e-3 | 1 | 2e-3 | 0.67 | stable | 20 | 0.04047 | 0 | 0.991 | 0.1523 | 0.0026 |
| lrscan_d7_L01_lr3e-3 | 1 | 3e-3 | 1.01 | stable | 20 | 0.04935 | 0 | 1.01 | 0.1501 | 0.0051 |
| lrscan_d7_L01_lr5e-3 | 1 | 5e-3 | 1.68 | diverged | 17 | 0.08800 | 7 | 1.552 | 0.2477 | 0.1417 |
| lrscan_d7_L10_lr1e-3 | 10 | 1e-3 | 0.34 | stable | 20 | 0.01345 | 0 | 0.981 | 0.8313 | 0.6018 |
| lrscan_d7_L10_lr2e-3 | 10 | 2e-3 | 0.67 | stable | 20 | 0.01670 | 0 | 1.01 | 0.7495 | 0.5689 |
| lrscan_d7_L10_lr3e-3 | 10 | 3e-3 | 1.01 | unstable | 20 | 0.03784 | 9 | 1.622 | 0.6977 | 0.5395 |
| lrscan_d7_L10_lr5e-3 | 10 | 5e-3 | 1.68 | diverged | 9 | 0.18040 | 5 | 4.128 | 0.6106 | 0.4801 |

Kills: L01 at 5e-3, epoch 17 (val p_L above 1.5 x its best 0.02107 for epochs 13-17); L10 at
5e-3, epoch 9 (above 1.5 x its best 0.01604 for epochs 5-9).

Sweep LR: 1e-3 (L=10 at 1e-3 stable). The conditional L=10 run at 5e-4 was not needed. L=2 is
`hl7_2x70` from the kd pilot, so `depth_d7_L02_u70` does not run.

Order of events: Part B started 2026-10-21 16:55 UTC, after the 3-hour wait for this amendment
ran out (access to the cluster lapsed between 10:23 and 19:00). This amendment was committed after
Part B started; it records Part A only and changes no Part B rule.
