<!-- Created: 2026-10-23 | Last modified: 2026-10-23 -->
# Pre-registration: deep narrow d=9 students (October 2026)

Committed before any run of this experiment trained. Code and run folders:
`experiments/d9_deep_students_2026-10/`.

## Question

At d=9 the kd pilot's narrow students failed: 3 x GRU 70 reached 5.86x MWPM and 1 x GRU 140
8.87x, while 3 x GRU 140 reached 1.05x. Both failures are at roughly 90k parameters, so the
question is whether the narrow stack fails for want of parameters or for want of depth. These
runs keep the width at 70 and stack 8 and 10 layers, which brings the parameter count up to
the d=9 budget without widening any layer.

The d=7 depth sweep (`prereg/depth_scaling_d7_2026-10.md`) found that at fixed width 70 the
stack beats a single layer of the same parameter count and keeps improving to 10 layers. Those
runs were hard label at d=7. This tests the same shape one distance up, with and without the
teacher.

## Runs

| run | student | parameters | alpha | teacher |
|---|---|---|---|---|
| kd9_10x70 | 10 x GRU 70 | 298,271 | 0 | f3280e240 |
| kd9_8x70 | 8 x GRU 70 | 239,051 | 0 | f3280e240 |
| hl9_10x70 | 10 x GRU 70 | 298,271 | 1 | none |

Parameter counts are asserted with `count_params()` before training; a mismatch stops the
experiment. All three are under the d=9 deployment budget of 303,521 (u280 at d=9), so
`--max-params 303521` stays set.

Comparisons already run, on the same pool, teacher and seed base: kd9_3x140 (329,001
parameters, 1.0461x), kd9_3x70 (91,001, 5.8586x), kd9_1x140 (92,961, 8.8682x).

## Fixed for every run

As kd9_3x140 in `prereg/` and `experiments/kd_pilot_p004_2026-10/`:

- d=9, r=9, p=0.004. GRU reset_after=False, stacked layers, then a Dense(1) logit.
- Teacher f3280e240 (3 x GRU 280, 1,246,001 parameters, epoch 240) for the alpha 0 runs,
  temperature 1. hl9_10x70 trains on hard labels only.
- Initialisation seed 0. Batch 10,000. Adam, global clipnorm 0.64. Peak LR 5e-4, warmup 3
  epochs, then cosine to 3e-5 over the epoch budget.
- 100M fresh Stim shots per epoch, fresh-shot seed base 5 x 10^12, the d9 base of the kd
  pilot, so these runs see the same shot stream as kd9_3x140, kd9_3x70 and kd9_1x140.
- TF32 in training as decided by the kd pilot d9 teacher check (`teacher_check/d9/
  tf32_decision.txt`: on). Every scored number is computed with TF32 off.
- Validation block FORMAL [15.0M, 15.2M): logged every epoch, used only by the kill rule.
- Epoch budget 240, the budget of the teacher. Exception, decided from the smoke test before
  any run trains: if kd9_10x70 projects past 120 h for 240 epochs with all three runs on the
  GPU, all three use 160 instead and the README records the measured rate.

## Scoring

- Every epoch from 40 is saved.
- Each saved epoch is scored on the selection block FORMAL [0, 10M), with MWPM decoded on the
  same shots.
- The selected checkpoint is the one with the lowest selection-block p_L, the later epoch on a
  tie (`select_checkpoint.py`).
- Only the selected checkpoint is scored on the eval block [15.2M, 17M). The sealed block and
  the held-out pool are not read.
- Convergence: slope of the selection-block ratio over the last 20 epochs; not converged if the
  slope is negative with t < -2.

## Kill rule

A run is killed when the training loss is NaN, or when the validation p_L stays above 1.5x the
run's best so far for 5 consecutive epochs. A killed run is reported as killed with its epoch,
and is not scored on the selection or eval blocks.

## McNemar tests

Paired on the eval block, b = the first decoder right and the second wrong, exact p:

- each run against MWPM on the same shots
- each run against the teacher f3280e240
- kd9_10x70 against hl9_10x70 (teacher against hard label at the same shape)
- kd9_10x70 against kd9_8x70 (two more layers)
- kd9_10x70 against kd9_3x140 (depth against width at a similar parameter count)

## Reading the result

One initialisation seed per run. Same-recipe seeds of the d9 teacher differ by about 0.02x
MWPM on the eval block (f3280 0.652x, f3280s1 0.633x), so differences smaller than about 3%
are within single-seed noise even when the paired McNemar p is small.

- Depth answers the d=9 failure if kd9_10x70 is at or below 1.05x, the ratio kd9_3x140 reached
  with more parameters in three wide layers.
- Parameter count, not depth, explains the earlier failure if kd9_10x70 and kd9_8x70 land near
  kd9_3x140 and the gap to kd9_3x70 tracks parameters rather than layers.
- Depth does not transfer from d=7 if both stacks stay above 2x while kd9_3x140 is at 1.05x.

## Resources

The GH200 is shared. Each start waits for GPU memory, host memory and free disk, as
`experiments/mwpm_fixes_d9_2026-10/driver.py` does. /home is a shared 1.4 TB volume with about
4 GB free at launch, and each run writes about 0.7 GB of epoch weights, so free space is
checked at launch and every 12 h; once a run is scored, its epoch weights other than the
selected one and the last are deleted and the deletion is logged.

## Amendments
