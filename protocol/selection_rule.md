# Checkpoint selection and held-out scoring

Created: 2026-10-14
Last modified: 2026-10-16

Committed before f9w_p003 finished training (it was at epoch 17 of 240) and before any of the
p = 0.003 runs was scored on an evaluation or held-out block.

## Scope

Every headline model at every (d, p): the p = 0.004 models (f5w_p004, f7w_p004, and the d9
3 x u280 runs f3280, f3280e240, f3280s1) and the p = 0.003 runs (f5w_p003, f7w_p003, f9w_p003) and the p = 0.005 runs (f5w_p005, f7w_p005, f9w_p005;
added 2026-10-15, before any of them was scored).

## Selection block

FORMAL [0, 10M) of the pool the model is scored on, with MWPM decoded on the same shots.

Only for models trained on fresh shots (`FRESH_SHOTS`, `NTRAIN=0`). Fixed-pool runs (arm b,
C1, D1, S1, S2, the R and K series, ...) trained on FORMAL [0, 15M), so this block is part of
their training data and they are not eligible.

Fresh shots cannot reproduce the FORMAL shots:

- A pool is one Stim measurement sampler, `compile_sampler(seed=gen_seed)`, with gen_seed 42,
  52, 60 or 61 for the FORMAL pools (`configs/partitions.yaml`). Fresh shots come from
  `compile_detector_sampler(seed=s)` with `s = seed_base + epoch * 10**6 + chunk` and
  seed_base 10^12, or 2 x 10^12 for f3280s1 (`src/fresh_shots.py`). So the sampler type differs,
  and so does every seed. `fresh_shots.check_seeds` stops a run if any seed listed in
  `partitions.yaml` falls in the run's seed range.
- An empirical check, in `experiments/selection_2026-10-14/check_fresh_vs_formal.py`. It took
  the first fresh chunk a run trains on (100k shots, epoch 0, chunk 0) and counted how many of
  those shots, compared on detection events plus logical flip, also occur anywhere in
  FORMAL [0, 10M). As a chance-level control, it made the same count for 100k pool shots from
  [10M, 10.1M), which are a different part of the same pool. Counts that match the control mean
  only common syndromes repeat, not the stream.

| pool | fresh seed base | fresh shots in [0, 10M) | control | chunk equal to rows 0.. row by row |
|---|---|---|---|---|
| d5 p0.003 seed 60 | 10^12 | 54,930 | 54,734 | 553 |
| d7 p0.003 seed 52 | 10^12 | 911 | 913 | 0 |
| d9 p0.003 seed 61 | 10^12 | 0 | 0 | 0 |
| d5 p0.004 seed 42 | 10^12 | 36,134 | 36,197 | 119 |
| d7 p0.004 seed 42 | 10^12 | 77 | 86 | 0 |
| d9 p0.004 seed 42 | 10^12 | 0 | 0 | 0 |
| d9 p0.004 seed 42 | 2 x 10^12 | 0 | 0 | 0 |

At d5 most shots have few or no detection events, so identical shots are common between any
two samples. The row-by-row matches at d5 are the same effect: 553 and 119 of 100k, against
the full 100k if the streams were equal.

## Rule

1. Score every saved checkpoint from epoch 40 to the last epoch on the selection block. Pick
   the lowest p_L; on a tie, the later epoch. No shortlist.
2. Score only the chosen checkpoint on the evaluation block FORMAL [15.2M, 17M). Report it with
   McNemar against MWPM (b = GRU right and MWPM wrong, c = the reverse, n_disc, exact p).
3. Never select on the evaluation or held-out blocks.

Per-checkpoint selection scores: `protocol/selection_scores.csv` (model, d, p, epoch, errors,
p_L, MWPM errors, ratio, weights path). Scorer:
`experiments/selection_2026-10-14/score_selection.py`.

Saved checkpoints:

- The p = 0.004 runs ended before this rule. Each kept only the validation-selected
  `.best.weights.h5` and the final `.lastepoch.weights.h5`, so those two are the candidates:

  | run | candidate epochs |
  |---|---|
  | f5w_p004 | 78, 80 |
  | f7w_p004 | 80 (best and last are the same epoch) |
  | f3280 | 148, 160 |
  | f3280e240 | 230, 240 |
  | f3280s1 | 147, 160 |

- From 2026-10-14 17:11 UTC, `experiments/ceiling_probe_d7_2026-10-05/snapshot_epochs.py`
  copies each epoch's weights of the p = 0.003 runs. The runs were not restarted.

  | run | epochs available |
  |---|---|
  | f5w_p003 | 53 to 80, plus the validation-best checkpoint if it is from epochs 40 to 52 |
  | f7w_p003 | 40 to 80 |
  | f9w_p003 | 40 to 240 |
  | f5w_p005, f7w_p005 | 40 to 80 (snapshots from the start of training, 2026-10-15) |
  | f9w_p005 | 40 to 240 (started 2026-10-16) |

Selected checkpoints and their evaluation-block scores (`selection_scores.csv`,
`eval_selected.csv`; McNemar b = GRU right and MWPM wrong, c = the reverse):

| run | candidate epochs | selected epoch | selection block | eval block | b | c | n_disc | exact p |
|---|---|---|---|---|---|---|---|---|
| f5w_p004 | 78, 80 | 80 | 0.6474x | 0.6503x | 7280 | 2515 | 9795 | <1e-300 |
| f7w_p004 | 80 | 80 | 0.6341x | 0.6315x | 4846 | 2000 | 6846 | 4.8e-267 |
| f3280 | 148, 160 | 160 | 0.6488x | 0.6516x | 2767 | 1345 | 4112 | 4.9e-111 |
| f3280e240 | 230, 240 | 240 | 0.6215x | 0.6231x | 2809 | 1271 | 4080 | 4.3e-131 |
| f3280s1 | 147, 160 | 160 | 0.6216x | 0.6332x | 2788 | 1291 | 4079 | 3.7e-124 |
| f5w_p003 | 53-80 | 78 | 0.6168x | 0.6251x | 3300 | 1100 | 4400 | 4.5e-252 |
| f7w_p003 | 40-80 | 75 | 0.6109x | 0.6180x | 1615 | 678 | 2293 | 1.5e-87 |
| f9w_p003 | 40-240 | 240 (tied with 236) | 0.8655x | 0.8837x | 660 | 548 | 1208 | 1.4e-3 |
| f5w_p005 | 40-80 | 78 | 0.6835x | 0.6783x | 13026 | 4891 | 17917 | <1e-300 |
| f7w_p005 | 40-80 | 80 | 0.6099x | 0.6073x | 11113 | 4027 | 15140 | <1e-300 |

f7w_p004's eval score is the launcher's, which scored the same epoch-80 checkpoint.

The launchers (run_arm8.sh, run_arm10.sh, run_arm11.sh) also score each run's
validation-best checkpoint on the evaluation block when training ends. Those scores were not
used for selection: f5w_p004 epoch 78 0.649x, f3280 epoch 148 0.665x, f3280e240 epoch 230
0.619x, f3280s1 epoch 147 0.641x, f5w_p003 epoch 80 0.623x, f7w_p003 epoch 70 0.635x,
f9w_p003 epoch 222 0.923x, f5w_p005 epoch 71 0.683x, f7w_p005 epoch 76 0.608x. f9w_p005 will get
the same extra score when it ends, since run_arm11.sh runs it.

## Held-out

One headline model per (d, p), declared here before any held-out scoring. Its checkpoint is
the one this rule selects. Each held-out block is scored exactly once, on that checkpoint
only.

| d | p | headline run | held-out block |
|---|---|---|---|
| 5 | 0.004 | f5w_p004 | HELDOUT seed 7001, 2M shots (the d5 FORMAL sealed block is spent) |
| 7 | 0.004 | f7w_p004 | FORMAL seed 42 sealed [17M, 19M) |
| 9 | 0.004 | f3280e240 | HELDOUT seed 7002, 4M shots |
| 5 | 0.003 | f5w_p003 | FORMAL seed 60 sealed [17M, 19M) |
| 7 | 0.003 | f7w_p003 | FORMAL seed 52 sealed [17M, 19M) |
| 9 | 0.003 | f9w_p003 | FORMAL seed 61 sealed [17M, 19M) |
| 5 | 0.005 | f5w_p005 | FORMAL seed 62 sealed [17M, 19M) |
| 7 | 0.005 | f7w_p005 | FORMAL seed 55 sealed [17M, 19M) |
| 9 | 0.005 | f9w_p005 | FORMAL seed 63 sealed [17M, 19M) |

The d9 p = 0.004 recipe (240 rather than 160 epochs) was chosen on 2026-10-13 by comparing
f3280e240 with f3280 on the evaluation block, before this rule existed. f3280 and f3280s1 are
scored on the selection block for the record. They are not headline candidates.

### Held-out size, d9 p = 0.003

The sealed block is 2M shots. MWPM on the selection block made 5,553 errors in 10M shots
(p_L 5.55e-4), so the expected count on 2M held-out shots is about 1,110 MWPM errors, and
about 670 for a GRU at 0.6x MWPM. That is above the 1,000-error threshold set for an extra
block, so no extra d9 p = 0.003 held-out block was generated.
