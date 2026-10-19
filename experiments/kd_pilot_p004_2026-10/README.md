# Distillation pilot, p = 0.004, FP32

Created: 2026-10-18
Last modified: 2026-10-19 (u70 e160 pair, launch.sh)

Matrix and rules fixed before any pilot run trained. FP32 only; separate from QAT and FPGA synthesis.

## Teachers

The selected checkpoints of `protocol/selection_rule.md`, rebuilt from `.weights.h5`
(`train_student.py --teacher-weights`, reset_after=False). Constants in `teachers.sh`.

| d | teacher | GRU | params | epoch | eval-block errors (logged) |
|---|---|---|---|---|---|
| 5 | f5w_p004 | 1 x 280 | 256,481 | 80 | 8,862 |
| 7 | f7w_p004 | 2 x 280 | 747,881 | 80 | 4,878 |
| 9 | f3280e240 | 3 x 280 | 1,246,001 | 240 | 2,543 |

Teacher check (`run_teacher_check.sh`, before any training at that d): parameter count, epoch,
sha256 of the copy in `teachers/`; eval block [15.2M, 17M) rescored with
`eval_student_on_tail.py --no-tf32` must give the logged error count; on [15.2M, 15.3M) the
training-step path (`fresh_shots._scatter` layout, `train_student.teacher_logit_column`) must give
the same features, the same decisions and the same error count as that rescoring.

TF32: the same training-step path with TF32 on over the whole eval block. If its error count is
within 5 shots of the `--no-tf32` count, training runs with TF32 on, otherwise off
(`teacher_check/d<D>/tf32_decision.txt`). Every reported number, students included, is scored
with TF32 off.

## Runs

reset_after=False, initialisation seed 0, 100M fresh Stim shots per epoch, batch 10,000, Adam
with global clipnorm 0.64, cosine to 3e-5. Distilled runs: BCE against the teacher's sigmoid
probability, alpha 0, T 1, teacher logits computed on every batch inside the training step,
teacher frozen in inference mode. Hard-label runs: alpha 1, no teacher, the same shots.

| d | peak LR | warmup | epochs | fresh-shot seed base |
|---|---|---|---|---|
| 5 | 3e-3 | none | 80 | 3 x 10^12 |
| 7 | 1e-3 | 3 | 80 | 4 x 10^12 |
| 9 | 5e-4 | 3 | 160 | 5 x 10^12 |

One seed base per d, shared by every run at that d (configs/partitions.yaml fresh_seed_ranges;
the teachers trained on base 10^12).

| run | student | params | compression | arms |
|---|---|---|---|---|
| kd5_u140, hl5_u140 | 1 x GRU 140 | 69,441 | 3.69x | distilled, hard label |
| kd5_u70, hl5_u70 | 1 x GRU 70 | 20,021 | 12.81x | distilled, hard label |
| kd5_u70_e160, hl5_u70_e160 | 1 x GRU 70 | 20,021 | 12.81x | distilled, hard label |
| kd7_2x140, hl7_2x140 | 2 x GRU 140 | 197,541 | 3.79x | distilled, hard label |
| kd7_2x70, hl7_2x70 | 2 x GRU 70 | 54,671 | 13.68x | distilled, hard label |
| kd7_1x113 | 1 x GRU 113 | 55,032 | 13.59x | distilled |
| kd9_3x140 | 3 x GRU 140 | 329,001 | 3.79x | distilled |
| kd9_3x70 | 3 x GRU 70 | 91,001 | 13.69x | distilled |
| kd9_1x140 | 1 x GRU 140 | 92,961 | 13.40x | distilled |

hl9_3x70 (pre-registered, conditional): runs only if kd9_3x70's selected checkpoint is between
1.0x and 2.3x MWPM on the selection block (`run_kd.sh` checks this).

Order: d5, then d7, then d9. After the d5 runs finish, results are reported before d7 starts.

The *_e160 pair was added after the d5 results (2026-10-19): every 80-epoch d5 run was still
improving at epoch 80. Identical to kd5_u70/hl5_u70 except cosine over 160 epochs; same seed
base, so epochs 1-80 see the same shots. Selection from epoch 40. Runs alongside d7.

## Checkpoint and reporting

- Every epoch from 40 on is saved and scored on the selection block FORMAL [0, 10M) with MWPM on
  the same shots (`score_selection.py`). Lowest p_L, the later epoch on a tie (`select_checkpoint.py`).
- Convergence: least-squares slope of the selection-block ratio over the last 20 epochs; a run is
  marked not converged when the slope is negative with t < -2.
- Only the selected checkpoint is scored on the eval block [15.2M, 17M). The sealed block
  [17M, 19M) and the held-out blocks (seeds 7001, 7002) are not read.
- `report.py --d <D>`: parameters, compression vs the teacher, selection- and eval-block p_L,
  ratio to MWPM and to the teacher on the same shots, McNemar vs MWPM, vs the teacher and (distilled
  runs) vs the hard-label control (b, c, n_disc, exact p; b = this run right, the other wrong),
  decision agreement with the teacher on all shots and on teacher p in (0.05, 0.95).
- One seed per run: same-recipe d9 teacher seeds differ by about 0.02x MWPM on the eval block, so
  smaller differences are within single-seed noise.

## Commands (grace1, from this folder)

    bash run_teacher_check.sh 5      # then 7, 9 before those distances train
    bash smoke.sh                    # timing runs, GPU otherwise idle
    bash launch_d5.sh                # the four d5 runs at once
    bash launch.sh kd5_u70_e160 hl5_u70_e160 kd7_2x140 hl7_2x140 kd7_2x70 hl7_2x70 kd7_1x113
    ~/tfenv/bin/python report.py --d 5
