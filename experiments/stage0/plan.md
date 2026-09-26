<!-- Created: 2026-09-26 | Last updated: 2026-09-26 -->
# Stage 0 plan: MWPM threshold for this repo's noise model

## Noise model (audited in step B1)

- Circuit: Stim `surface_code:rotated_memory_z`, built only by `src/circuit_generators.build_circuit`.
- Memory basis Z, one logical observable. Rounds r = d.
- Noise, all equal to p: `after_clifford_depolarization` (DEPOLARIZE1 after 1-qubit Cliffords,
  DEPOLARIZE2 after CNOTs), `before_round_data_depolarization` (DEPOLARIZE1 on data qubits each
  round), `after_reset_flip_probability` and `before_measure_flip_probability` (X_ERROR).
  No other noise (no idling noise beyond the per-round data depolarization).
- At p = 0.004: d=5 has 64 qubits, 120 detectors, 1,953 DEM error mechanisms;
  d=7: 118 / 336 / 6,602; d=9: 188 / 720 / 14,966.
- The MWPM baseline everywhere in `src/` (mwpm_on_block, eval_student_on_tail, the distillation
  scripts) decodes `build_circuit(d, p, d).detector_error_model(decompose_errors=True)` with
  PyMatching. The sweep gives sinter that same DEM for every task.

## Sweep

- p grid: 0.003, 0.004, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.008, 0.009
- d = 5, 7, 9; r = d
- Decoder: pymatching through `sinter.collect`
- Stopping rule per point: 1,000 logical errors or 10^7 shots, whichever first
- Shots are fresh sinter samples; no FORMAL pool (and so no val, eval or sealed shot) is read.
- Workers: 64 of 72 cores. At launch the load average was ~2 (sglang scheduler, the A7 smoke run),
  so 64 leaves headroom for those and for the Phase C checks.

## Runtime estimate (timing run, 200,000 shots per point, 64 workers)

Per-shot decode cost 1.2-4.5 us (d=5), 3.8-13.3 us (d=7), 8.6-34 us (d=9) per core. The largest
point under the stopping rule is d=9, p=0.003 (LER ~5.6e-4, so ~1.8M shots, ~15 core-seconds).
Whole sweep ~100 core-seconds: under a minute of wall time. Timing data:
`experiments/stage0/timing_20260926T052106Z/`.

## Analysis (src/threshold_analysis.py)

- LER per point = errors / shots (per shot, over all r = d rounds), 95% Wilson interval.
- Plot LER vs p for each d on log-log axes.
- Crossings: for each pair (5,7) and (7,9) separately, find the grid interval where
  LER_small - LER_large changes sign and interpolate linearly in (log p, log LER).
- Also reported, secondary: the same crossings using per-round LER,
  p_round = (1 - (1 - 2 LER)^(1/r)) / 2, since per-shot LER at r = d mixes distance and duration.
- Compared with Fowler's 0.57%. Note that the noise model here is Stim's uniform depolarizing
  model, not Fowler's, so the two numbers need not agree.

## Pre-registered rule

A Stage 4 physical error rate p is usable only if it lies clearly below both crossings
(d=5/7 and d=7/9). What counts as "clearly below", and which p values Stage 4 uses, is
Karrie's decision after B4; this plan does not change them.
