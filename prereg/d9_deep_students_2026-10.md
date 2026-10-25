<!-- Created: 2026-10-23 | Last modified: 2026-10-25 -->
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

### 2026-10-24: kd9_taper4 added, the u70 stacks paused

The three u70-stack runs were paused at an epoch boundary after epoch 52-61 of 160, by
decision and not by any rule in this document. At matched epochs on the validation block they
sat near 5x MWPM while kd9_3x140 was at 1.5x, so the question this experiment asks, whether
depth at width 70 recovers the d=9 result, was already answered for practical purposes: it
does not. Their weights, Adam state and per-epoch history are kept, and they can be resumed
exactly (`resume_student.sh`); nothing about their recipe or budget changes.

One run is added, launched after this amendment is committed:

| run | student | parameters | alpha | teacher |
|---|---|---|---|---|
| kd9_taper4 | GRU 160-110-90-70 | 293,261 | 0 | f3280e240 |

The tapered stack spends the d=9 budget on a wide first layer, which reads the 80 detector
inputs per round, and narrows with depth. It is the shape between kd9_3x140 (329,001
parameters, 1.0461x, over the budget) and the uniform u70 stacks (under the budget, near 5x
mid-training). The parameter count is asserted with `count_params()` before training and is
under the d=9 cap of 303,521.

Everything else follows kd9_3x140 and the sections above: teacher f3280e240 with alpha 0 and
temperature 1, initialisation seed 0, batch 10,000, Adam with global clipnorm 0.64, peak LR
5e-4, warmup 3 epochs, cosine to 3e-5 over 160 epochs, 100M fresh shots per epoch, seed base
5 x 10^12, TF32 on in training and off in scoring, every epoch saved from 40, the same
selection rule, the same kill rule and the same convergence test.

Progress reporting: every 10 epochs the validation p_L of kd9_taper4 and of kd9_3x140 at that
same epoch are appended to `status.log`, so the two trajectories can be compared while the run
is going. These are monitoring lines; no decision is taken on them.

McNemar on the eval block for kd9_taper4: against MWPM on the same shots, against the teacher
f3280e240, and against kd9_3x140.

### 2026-10-25: kd9_taper3_t100 added, kd9_taper4 paused

kd9_taper4 was paused at an epoch boundary after epoch 120 of 160, by decision and not by any
rule in this document. Its trajectory was already decided: 2.87x MWPM on the validation block at
epoch 110 against 1.10x for kd9_3x140 at the same epoch, improving about 0.1x per 10 epochs, so
the remaining 40 epochs cannot bring it near kd9_3x140. Weights, Adam state and per-epoch history
are kept and the run is exactly resumable (`resume_student.sh`); its recipe and 160-epoch budget
do not change. It is resumed for its remaining epochs once kd9_taper3_t100 finishes, and then
scored.

One run is added, launched after this amendment is committed, alone on the GPU:

| run | student | parameters | alpha | teacher |
|---|---|---|---|---|
| kd9_taper3_t100 | GRU 160-120-100 | 283,241 | 0 | f3280e240 |

Three layers at a parameter count close to kd9_taper4 and kd9_3x140 (283,241 against 293,261 and
329,001), tapering but staying wide: the last layer is 100 rather than 70. It separates the two
things that changed between kd9_3x140 (1.05x) and kd9_taper4 (near 2.9x mid-training), the number
of layers and the width of the later layers, by holding the depth at three and narrowing only
mildly. The parameter count is asserted with `count_params()` before training and is under the
d=9 cap of 303,521.

Everything else follows kd9_3x140 and the sections above: teacher f3280e240 with alpha 0 and
temperature 1, initialisation seed 0, batch 10,000, Adam with global clipnorm 0.64, peak LR 5e-4,
warmup 3 epochs, cosine to 3e-5 over 160 epochs, 100M fresh shots per epoch, seed base 5 x 10^12,
TF32 on in training and off in scoring, every epoch saved from 40, the same selection rule, the
same kill rule and the same convergence test.

Progress reporting: every 10 epochs the validation p_L of kd9_taper3_t100, kd9_3x140 and
kd9_taper4 at that same epoch are appended to `status.log`. These are monitoring lines; no
decision is taken on them.

McNemar on the eval block for kd9_taper3_t100: against MWPM on the same shots, against the
teacher f3280e240, against kd9_3x140 and against kd9_taper4.

### 2026-10-25: kd9_3x134 added, kd9_taper3_t100 paused

kd9_taper3_t100 was paused at the end of epoch 40 of 160, by decision and not by any rule in this
document. It tracked kd9_taper4 to within 1% at every matched epoch (3.98x against 3.92x MWPM on
the validation block at epoch 39) while kd9_3x140 was at 1.82x, and train loss tracked validation
loss to about 1% for both tapers, so the deficit is on the training objective and the two tapered
shapes behave as one. Weights, Adam state and history are kept and the run is exactly resumable
(`resume_student.sh`); its recipe and 160-epoch budget do not change.

One run is added, launched after this amendment is committed:

| run | student | parameters | alpha | teacher | epochs |
|---|---|---|---|---|---|
| kd9_3x134 | 3 x GRU 134 | 302,841 | 0 | f3280e240 | 240 |

kd9_3x140 reached 1.0461x but at 329,001 parameters, over the d=9 deployment budget of 303,521.
A uniform 3 x GRU 134 is the widest three-layer stack that fits the budget, 302,841 parameters,
and it isolates the one factor the tapers left confounded: at depth 3 and the same budget, a
uniform width against a taper (kd9_taper3_t100, 160-120-100, 283,241). If it lands near 1.05x the
d=9 result holds inside the budget and the taper was the problem; if it lands near 2x the
remaining gap to kd9_3x140 is the last 26,160 parameters of width.

The epoch budget is 240, the budget of the teacher and of kd9_3x140, rather than the 160 the other
runs of this experiment use, so the comparison against kd9_3x140 is at matched epochs over the
whole schedule. At the measured rate for a three-layer stack of this width this is about 33 h,
inside the 120 h limit of the Runs section.

Everything else follows kd9_3x140 and the sections above: teacher f3280e240 with alpha 0 and
temperature 1, initialisation seed 0, batch 10,000, Adam with global clipnorm 0.64, peak LR 5e-4,
warmup 3 epochs, cosine to 3e-5 over the 240 epochs, 100M fresh shots per epoch, seed base
5 x 10^12, TF32 on in training and off in scoring, every epoch saved from 40, the same selection
rule, the same kill rule and the same convergence test.

Progress reporting: every 10 epochs the validation p_L of kd9_3x134, kd9_3x140, kd9_taper4 and
kd9_taper3_t100 at that same epoch are appended to `status.log`, as integer error counts over the
200,000-shot validation block divided by the block size. These are monitoring lines; no decision
is taken on them.

McNemar on the eval block for kd9_3x134: against MWPM on the same shots, against the teacher
f3280e240, against kd9_3x140 and against kd9_taper3_t100.

### 2026-10-25: correction to the kd9_3x134 epoch budget

The amendment above set kd9_3x134 to 240 epochs on the statement that 240 is the budget of
kd9_3x140. That is wrong: kd9_3x140 trained for 160 epochs (`EPOCHS=160` in its env.txt, 160
validation epochs in its log). Only the teacher f3280e240 ran 240 epochs, from fresh-shot seed
base 1 x 10^12, which `configs/partitions.yaml` reserves for 240.

The first launch stopped before training on the seed-range guard in `src/fresh_shots.py`: base
5 x 10^12 is reserved for 160 epochs, so 240 epochs from it would have drawn seeds outside the
reservation. No weights were written and nothing was scored (run folder
`runs/kd9_3x134_20261025T202323Z`, kept).

kd9_3x134 runs for 160 epochs, the budget of every other run in this experiment and of
kd9_3x140, on the same seed base and so the same shot stream. Nothing else about the run changes.
Extending the reservation to 240 was rejected: it would have given kd9_3x134 a cosine schedule
over a different horizon from the run it exists to isolate.
