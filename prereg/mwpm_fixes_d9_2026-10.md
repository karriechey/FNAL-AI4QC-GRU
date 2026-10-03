<!-- Created: 2026-10-03 | Last updated: 2026-10-03 -->
# Pre-registration: d=9 runs aimed at beating MWPM (October 2026)

Written before any of these runs started. The queue is run by
`experiments/mwpm_fixes_d9_2026-10/driver.py` from `jobs.tsv` in the same folder; this
file and `jobs.tsv` hold the same table.

## Starting point

- **d=9 best so far:** R02 (u280, cosine 3e-3 to 3e-5, 100M, 80 epochs). p_L / MWPM = 5.02 on the evaluation block. Its best checkpoint was at epoch 79 of 80.
- **Longer runs diverged early, while the LR was near the peak:**
  - R17 (160 epochs): best epoch 11, 7.28.
  - R16 (200M): best epoch 5, 7.58.
- **Width barely matters at d=9:** u200 gave 5.12 and u280 gave 5.02.
- **Reading:** d=9 is limited by training (undertrained and unstable at a 3e-3 peak), not by capacity.

## Fixed for every run

- **Problem:** d = 9, r = 9, p = 0.004.
- **Model:** GRU with reset_after = False, then a Dense(1) logit.
- **Optimizer:** cosine LR to 3e-5. Batch 10,000 and seed 0 unless the row says otherwise. Early stopping off.
- **Data:** the FORMAL pool `data_d9_p0.004_r9_FORMAL.npz` (generation seed 42) supplies the training prefix [0, 15M), validation [15.0M, 15.2M) and evaluation [15.2M, 17M). Training-only pools are added after the prefix, in this order:
  - **100M:** seeds 48 (35M) and 49 (50M), as in R02.
  - **200M:** seeds 48, 49, 58 and 59 (35M + 3 × 50M), as in R16.
- **Checkpoint:** the epoch with the fewest validation errors (tiebreak: lower validation loss). With EMA, the choice is made on, and saves, the EMA weights.
- **Score:** that checkpoint on the evaluation block, with MWPM decoded on the same 1.8M shots (`eval_student_on_tail.py`). McNemar b (GRU-only correct), c (MWPM-only correct), n_discordant and exact p are reported for every run. No Fisher combination across seeds.
- **Sealed block [17M, 19M):** not read.
- **Parameter budget:** at most 303,521 parameters, the single-layer u280 count at d=9. The launcher (`MAX_PARAMS`) and `train_student.py --max-params` refuse anything larger.
- **One run on the GPU at a time.** Before each launch, the GPU must have no other user's compute process; otherwise the driver waits and re-checks every 15 minutes.
- **Code:** branch `mwpm-fixes`. Every new setting is off by default, and with all of them off a run reproduces the earlier losses exactly (`tests/knob_checks.py defaults_off`).

## Runs (launch order)

STAB means the value comes from the Tier A promotion (see the promotion rule below).

| # | Tier | Peak LR | Warmup (epochs) | Clipnorm | Epochs | Batch | Data | Other | Why |
|---|---|---|---|---|---|---|---|---|---|
| 9-1 | A | 1e-3 | 3 | 1.0 | 160 | 10,000 | 100M | | lower peak, warmup and clip address the early divergence; 160 epochs address undertraining |
| 9-2 | A | 1.5e-3 | 3 | 1.0 | 160 | 10,000 | 100M | | brackets the LR |
| 9-3 | A | 3e-3 | 3 | 1.0 | 160 | 10,000 | 100M | | do warmup and clip alone fix the known peak? |
| 9-4 | B | STAB | STAB | STAB | 160 | 10,000 | 100M | warm start from R01 (d7 u280 200M, 1.060). Recurrent kernel, GRU bias and head are loaded; the input kernel (48 → 80 positions) starts fresh. | curriculum from a near-MWPM decoder |
| 9-5 | B | STAB | STAB | STAB | 160 | 10,000 | 200M | | more data once stable |
| 9-6 | B | STAB | STAB | STAB | 240 | 10,000 | 100M | only if the promoted run's best epoch is in the last 10% of its epochs; otherwise skipped, with the reason logged | longer only if still improving |
| 9-7 | B | STAB | STAB | STAB | 160 | 10,000 | 100M | EMA of weights, decay 0.999; the EMA weights are checkpointed and scored | smooths the end of training |
| 9-8 | B | STAB | STAB | STAB | 160 | 10,000 | 100M | 2-layer stacked GRU within the 303,521 budget | depth instead of width. **Not runnable yet:** needs the stacked GRU, including the evaluation script rebuilding two layers. Pending approval; skipped by the driver until present. |
| 9-9 | B | STAB / √2 | STAB | STAB | 160 | 5,000 | 100M | | more updates per epoch |
| 9-10 | B | best so far | | | | | | seed 1 | replicate the best config among 9-1 to 9-9 |

## Rules

- **Win:** p_L / MWPM on the evaluation block < 1.0 *and* McNemar exact p < 0.01.
- **Improvement:** the ratio is below the current d=9 best: 5.019 (R02), or the best of this queue so far.
- **Kill (saves GPU time):**
  - Training loss NaN: the run is stopped and recorded as diverged, with the epoch.
  - At 25% of the run's epochs (epoch 40 of 160, epoch 60 of 240): if validation p_L exceeds 1.25 × R02's validation p_L at 25% of its 80 epochs (epoch 20), the run is stopped and recorded as `Killed @ ep N`.
  - The threshold is read from R02's `history.json` when the driver starts and is printed on its first log line. Comparing at the same fraction of the schedule keeps the lower-peak and longer runs from being stopped just because their LR is lower early on.
- **Promotion:**
  - After 9-1, 9-2 and 9-3 finish, the one with the lowest p_L / MWPM sets the peak LR, warmup and clipnorm of every row marked STAB (written to `stability.json`).
  - Runs that were stopped don't count.
  - If all three were stopped, the queue stops there and waits.
- **9-10 placement:** last, unless some run wins. Then it runs next, replicating the best config so far.
- **Deliberately absent:** no d=7 runs and no MLP variants in this queue.

## Report at the end of the queue

- One row per run (ratio, McNemar p, kept / killed / skipped, one-line reading).
- Which settings helped, the best config, and whether d=9 got within 2× of MWPM.
- Where the best run's checkpoint landed: still improving at the end, or plateaued. That decides whether the deferred MLP runs are worth testing.
