<!-- Created: 2026-10-29 | Last modified: 2026-10-29 -->
# X-basis memory at p = 0.004

Fixed before any X-basis run trained.

## Question

Do the Z-basis teachers' recipes give the same GRU / MWPM ratio on X-basis memory?

## Circuit

`circuit_generators.build_circuit(d, p, rounds, basis=None)`: `basis='x'` builds Stim's
`surface_code:rotated_memory_x` with the same four noise channels at p. With no argument the
basis comes from `QEC_BASIS`, else Z, so every existing caller is unchanged. The X launchers set
`QEC_BASIS=x` in every container (fresh shots, detector layout, MWPM decoding).
`partitions.check_pool_geometry` (training and eval) and `score_selection.py` stop when a pool's
basis (fingerprint `basis`, or `_X_` in the file name) differs from `QEC_BASIS`.

## Pools

New X-basis FORMAL pools, 19M shots each in the configs/partitions.yaml layout, generation seeds
64 / 65 / 66 for d5 / d7 / d9 (registered there). The eval pool is each pool's eval block
[15.2M, 17M); the selection block is [0, 10M) and the validation block [15.0M, 15.2M). The
sealed block [17M, 19M) is reserved and never scored.

/home has 2.8 GB free, so the pools live in /tmp/kchey_xbasis/pools and store detection events
and flips only (no raw measurements): about 2.3 / 6.4 / 13.7 GB instead of 5 / 13 / 27 GB.
The GRU reads detection events only, and the readers skip measurements when a pool has none.
The queue touches /tmp/kchey_xbasis daily against the 10-day /tmp cleanup. The pools can be
regenerated from their seeds on grace1.

## Runs

From scratch (no Z weights), same recipe as the Z teacher, X-basis fresh shots:

| run | d | model | parameters | Z teacher | peak LR | warmup | epochs | fresh seed base |
|---|---|---|---|---|---|---|---|---|
| x5_u280 | 5 | u280 | 256,481 | f5w_p004 (0.650x) | 3e-3 | 0 | 80 | 12e12 |
| x7_2xu280 | 7 | 2 x u280 | 747,881 | f7w_p004 (0.632x) | 1e-3 | 3 | 80 | 13e12 |
| x9_3xu280 | 9 | 3 x u280 | 1,246,001 | f3280e240 (0.623x) | 5e-4 | 3 | 240 | 14e12 |

Common: 100M fresh shots per epoch, cosine to 3e-5, global clipnorm 0.64, batch 10,000, Adam,
reset_after=False, init seed 0, TF32 on in training and off in scoring. Kill rule as
depth_vs_distance (NaN, or validation p_L above 1.5 x its best for 5 epochs), as a safety net.

Estimated hours, from the Z teachers' runtimes: 5 / 9 / 43 h plus scoring. The d9 run sets the
length, about 2 days.

## Scoring

Every epoch from 40 scored on the selection block with MWPM on the same shots; the checkpoint
is the lowest selection-block p_L (later epoch on a tie), the Z teachers' criterion. The Z
teachers could only choose among the epochs they had saved (best-validation and last), so the X
runs choose from a larger set; the selected epochs are reported side by side. The selected
checkpoint is scored on the eval block, GRU vs MWPM on identical X-basis shots, exact McNemar.

## Z vs X

Different shots, so unpaired: per d, GRU / MWPM for Z and X with 95% intervals from a
multinomial bootstrap of each run's paired 2x2 table (Z counts from
GRU_error_rate_runs_2026-10-20.csv), the X - Z difference with its interval, and the MWPM p_L
in each basis. Output: zx_comparison.md / .csv.

## Execution

    setsid nohup bash queue_x.sh > queue_x.out 2>&1 < /dev/null &

Pools d5, d7, d9 in turn; each teacher starts once its pool exists. When all three are
finished: report_x.py (GRU_xbasis_runs.csv, zx_comparison), then the depth_vs_distance queue,
paused with its STOP file at 2026-10-29 04:00 UTC, is resumed. The GRU_xbasis_runs sheet tab is
filled from GRU_xbasis_runs.csv when synced from the Mac.
