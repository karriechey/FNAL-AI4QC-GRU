<!-- Created: 2026-10-03 | Last updated: 2026-10-05 (amendment 6) -->
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
- **One run on the GPU at a time.** Before each launch, if another user's process is on the GPU, the driver launches only if it holds < 10 GB and GPU utilization stays near 0% over a 5-minute check (mean < 5%, max < 20%). Otherwise it waits and re-checks every 15 minutes, adding the waiting time to `blocked.json`.
- **Code:** branch `mwpm-fixes`. Every new setting is off by default, and with all of them off a run reproduces the earlier losses exactly (`tests/knob_checks.py defaults_off`).

## Runs (launch order)

STAB means the value comes from the Tier A promotion (see the promotion rule below).

| # | Tier | Peak LR | Warmup (epochs) | Clipnorm | Epochs | Batch | Data | Other | Why |
|---|---|---|---|---|---|---|---|---|---|
| 9-1 | A | 1e-3 | 3 | 0.64 | 160 | 10,000 | 100M | | lower peak and warmup address the early divergence; 160 epochs address undertraining |
| 9-2 | A | 1.5e-3 | 3 | 0.64 | 160 | 10,000 | 100M | | brackets the LR |
| 9-3 | A | 3e-3 | 3 | 0.64 | 160 | 10,000 | 100M | | does warmup alone fix the known peak? |
| 9-4 | B | STAB | STAB | STAB | 160 | 10,000 | 100M | warm start from R01 (d7 u280 200M, 1.060). Recurrent kernel, GRU bias and head are loaded; the input kernel (48 → 80 positions) starts fresh. | curriculum from a near-MWPM decoder |
| 9-5 | B | STAB | STAB | STAB | 160 | 10,000 | 200M | | more data once stable |
| 9-6 | B | STAB | STAB | STAB | 240 | 10,000 | 100M | only if the promoted run's best epoch is in the last 10% of its epochs; otherwise skipped, with the reason logged | longer only if still improving |
| 9-7 | B | STAB | STAB | STAB | 160 | 10,000 | 100M | EMA of weights, decay 0.999; the EMA weights are checkpointed and scored | smooths the end of training |
| 9-11 | A | 5e-4 | 3 | 0.64 | 160 | 10,000 | 100M | added by amendment 4; runs after 9-5 | is the best peak LR below 1e-3? |
| 9-9 | B | STAB | STAB | STAB | 160 | 5,000 | 100M | amended: no ÷√2 | more updates per epoch; only the batch changes |
| 9-10 | B | best so far | | | | | | seed 1 | replicate the best config among 9-1 to 9-9 |

## Rules

- **Win:** p_L / MWPM on the evaluation block < 1.0 *and* McNemar exact p < 0.01.
- **Improvement:** the ratio is below the current d=9 best: 5.019 (R02), or the best of this queue so far.
- **Kill (saves GPU time):** a run is stopped, and recorded with the epoch, when any of these holds:
  1. Training loss is NaN (recorded as diverged).
  2. Validation p_L stays above 1.5 × the run's own best validation p_L so far for 5 consecutive epochs (`Killed @ ep N`).
  3. Validation p_L at epoch 60 is above 2 × R02's validation p_L at epoch 60 (`Killed @ ep 60`). R02's value is read from its `history.json` when the driver starts and printed on its first log line.
- **Promotion:**
  - After 9-1, 9-2 and 9-3 finish, the one with the lowest p_L / MWPM sets the peak LR, warmup and clipnorm of every row marked STAB (written to `stability.json`).
  - Runs that were stopped don't count.
  - If all three were stopped, the queue stops there and waits.
- **9-10 placement:** last, unless some run wins. Then it runs next, replicating the best config so far among the remaining runs (9-1 to 9-9 without 9-8).
- **Deliberately absent:** no d=7 runs in this queue.

## Amendments (2026-10-03, before any run started; no data from this queue seen)

1. **Tier A clipnorm 0.64 instead of 1.0.** R02, R16 and R17 already clipped at 0.64, so 1.0 would have relaxed clipping rather than added it. At 0.64, Tier A changes only the peak LR and the warmup relative to R02.
2. **Kill rule replaced.** The earlier rule (epoch-20, or 25%-of-schedule, comparison against R02 at 1.25×) is replaced by the three conditions above. Reason: a lower peak LR with warmup is expected to trail R02 early, so an early comparison would stop exactly the runs meant to fix the divergence. The new rule catches a run that degrades and stays degraded (5 epochs at 1.5× its own best), and one that is far behind R02 well into training (2× at epoch 60).
3. **9-8 (stacked GRU) declined.** It needed the evaluation script to rebuild two layers, and the evaluation path is not changed in this queue. Moved to Deferred.

4. **2026-10-04, after 9-1 to 9-4 and during 9-5 (results seen; recorded as such).**
   - **What we'd seen:** Tier A gave 1e-3: 2.54×, 1.5e-3: 2.83×, 3e-3: killed at epoch 43. The best peak LR sat at the low edge of the bracket.
   - **9-11 added:** peak LR 5e-4, otherwise as 9-1. It is a Tier A row and runs right after 9-5.
   - **Promotion redone:** after 9-11, over 9-1, 9-2 and 9-11. The first promotion (9-1) is kept in `stability_tierA_2026-10-04.json`. Rows 9-4 and 9-5 had already run with 9-1's settings.
   - **9-9 changed:** batch 5,000 at the promoted peak LR, without the ÷√2. As first written it changed the LR and the batch together, so a difference couldn't be attributed to either.
   - **Reason:** close the LR bracket before spending the remaining runs on a peak that may not be best.
   - Because this amendment follows seen results, 9-11 and anything chosen through the redone promotion are exploratory relative to the original pre-registration.

5. **2026-10-04, during 9-5: two runs at once.** The one-run-at-a-time rule is lifted for 9-11, which starts now alongside 9-5 (GPU about 40% busy, each job capped at 4 GB, host memory sufficient). It is started by `run_one.py` through the same code path as the queue, with the same kill rules, logging and results row. 9-7 now comes before 9-6 in `jobs.tsv`, so if 9-5 ends before 9-11 the driver waits for the redone promotion instead of skipping 9-6. Concurrency changes run times, not results (determinism is per process).

6. **2026-10-05: concurrency throughout; pauses for the d7 ceiling probe.**
   - **Concurrent rows:** the driver now starts rows concurrently whenever host memory allows: at least `need_host_gb` free, no queue run still loading, and swap not growing. The GPU is not shared with other users, so the GPU-sharing wait is removed.
   - **9-10:** still waits until every other row has finished, unless a run wins.
   - **Pauses:** 9-11 (after epoch 68) and 9-5 were paused at epoch boundaries to free memory for the d7 ceiling probe (diagnostic, outside this queue). They resume from their last-epoch weights and optimizer state, which carries the update count and so the LR-schedule step. Their histories are rebuilt from the run logs, as for the earlier paused runs. The kill rules see the epochs from before the pause.
   - Run times change; results do not (determinism is per process).

## Deferred (not implemented, not run)

Decided after the d=9 results.
- **9-8: 2-layer stacked GRU within the 303,521 budget** (e.g. two layers of 160, 269,921 parameters). Needs `--units` to take a list in both `train_student.py` and `eval_student_on_tail.py`.
- **MLP head:** final GRU state → Dense(128, ReLU) → Dense(1). u256 + head: 291,841 parameters at d=9. Paired with a plain u256 control.
- **MLP encoder:** each round's detector vector → shared Dense(128, ReLU) → GRU u248 (291,105 parameters at d=9). Adds per-round FPGA latency; lower priority.

## Report at the end of the queue

- One row per run (ratio, McNemar p, kept / killed / skipped, one-line reading).
- Which settings helped, the best config, and whether d=9 got within 2× of MWPM.
- Where the best run's checkpoint landed: still improving at the end, or plateaued. That decides whether the deferred MLP runs are worth testing.
