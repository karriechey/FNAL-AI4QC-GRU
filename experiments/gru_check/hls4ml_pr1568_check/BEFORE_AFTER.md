<!-- Created: 2026-09-27 | Last updated: 2026-09-27 -->
# hls4ml PR #1568: GRU `reset_after=False`, before vs after

Trained d=5 GRU decoder (u140, `reset_after=False`, run of record, seed 0; checkpoint `gru_d5_p004_u140_hard_seed0_ntr10000000_b10000.best.weights.h5`), converted with the Vivado backend, `io_parallel`, every precision `ap_fixed<32,16>`, GRU activation tables of 65,536 entries (hls4ml default input ranges). Reference: the same Keras model on the CPU (no TF32). Inputs: first 10000 of the d=5 eval block [15200000, ...) (10,000 shots), identical for both versions.

- before: upstream hls4ml `1.3.0`
- after: hls4ml `1.3.0.post1568+ge1721b302f8a29397b532b82f21dc8d908781a70` (PR #1568, commit `e1721b3`)

| hls4ml | max \|Δlogit\| | mean \|Δlogit\| | decision agreement | decisions that differ | generated `reset_after` |
|---|---|---|---|---|---|
| 1.3.0 (before) | 26.528 | 8.9426 | 65.39 % | 3,461 / 10,000 | `not set` |
| PR #1568 (after) | 0.220 | 0.0128 | 100.00 % | 0 / 10,000 | `false` |

Decision = sign of the logit (logit > 0 predicts a logical flip). "Generated `reset_after`" is read from the generated `firmware/parameters.h`; hls4ml 1.3.0 writes no such setting, and its GRU kernel always computes the `reset_after=True` form. Keras logit standard deviation on these shots: 7.97.

## Supporting checks (same shots and settings; not part of the main comparison)

| model | hls4ml | max \|Δlogit\| | mean \|Δlogit\| | decision agreement | decisions that differ | generated `reset_after` |
|---|---|---|---|---|---|---|
| u140 reset after False 20M aug seed0 | 1.3.0 | 23.753 | 8.3937 | 61.65 % | 3,835 / 10,000 | `not set` |
| u140 reset after False 20M aug seed0 | PR #1568 | 0.113 | 0.0106 | 99.99 % | 1 / 10,000 | `false` |
| u70 reset after False gkd20m hard seed0 | 1.3.0 | 45.633 | 17.5870 | 20.58 % | 7,942 / 10,000 | `not set` |
| u70 reset after False gkd20m hard seed0 | PR #1568 | 0.127 | 0.0185 | 99.99 % | 1 / 10,000 | `false` |
| u140 reset after True unroll 20M aug seed0 | 1.3.0 | 0.148 | 0.0452 | 99.99 % | 1 / 10,000 | `not set` |
| u140 reset after True unroll 20M aug seed0 | PR #1568 | 0.148 | 0.0452 | 99.99 % | 1 / 10,000 | `true` |

Figure: `before_after_decision_agreement.png` / `.pdf`.
Source runs: `run_20260927T061602Z_upstream_1.3.0` (before), `run_20260927T055605Z` (after).
