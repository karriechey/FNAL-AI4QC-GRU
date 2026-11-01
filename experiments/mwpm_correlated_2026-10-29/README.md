<!-- Created: 2026-10-29 | Last updated: 2026-10-29 -->
# Correlated MWPM vs uncorrelated MWPM vs the best GRU, p = 0.004, d = 5, 7, 9

Question: does the GRU's advantage over standard MWPM (about 35% lower pL) hold against correlation-aware
MWPM, and does that change with distance?

## Setup

- Shots: the FORMAL eval block [15.2M, 17M) of each distance's pool, 1,800,000 shots, read from the
  existing per-shot dumps of the best GRU at each distance
  (`kd_pilot_p004_2026-10/teacher_check/d{5,7,9}/teacher_eval_per_shot.npz` on grace1: detection events,
  truth, GRU decisions). No data generated, no model run, nothing on the GPU, no existing file changed.
- GRU per distance (the selected checkpoints of `protocol/selection_rule.md`): f5w_p004 (1 x GRU 280,
  256,481 params), f7w_p004 (2 x 280, 747,881), f3280e240 (3 x 280, 1,246,001); reset_after=False.
- MWPM: `build_circuit(d, 0.004, d).detector_error_model(decompose_errors=True)`, PyMatching 2.4.0,
  stim 1.16.0. Uncorrelated: defaults. Correlated: `enable_correlations=True` in both
  `Matching.from_detector_error_model` and `decode_batch`. The recomputed uncorrelated decisions equal the
  stored MWPM decisions shot for shot at every distance (same DEM).
- McNemar: exact two-sided binomial on discordant shots; b = first decoder right / second wrong.

## Result (1,800,000 shots each)

| d | MWPM pL | correlated MWPM pL | GRU pL | GRU x MWPM | GRU x corr. MWPM | corr. x MWPM |
|---|---|---|---|---|---|---|
| 5 | 0.007571 (13,627) | 0.005714 (10,286) | 0.004923 (8,862) | 0.650 | 0.862 | 0.755 |
| 7 | 0.004291 (7,724) | 0.002383 (4,290) | 0.002710 (4,878) | 0.632 | 1.137 | 0.555 |
| 9 | 0.002267 (4,081) | 0.000891 (1,604) | 0.001413 (2,543) | 0.623 | 1.585 | 0.393 |

Relative improvement (1 - pL_A / pL_B):

| d | GRU vs MWPM | GRU vs correlated MWPM | correlated MWPM vs MWPM |
|---|---|---|---|
| 5 | +35.0% | +13.8% | +24.5% |
| 7 | +36.8% | -13.7% | +44.5% |
| 9 | +37.7% | -58.5% | +60.7% |

McNemar, GRU vs correlated MWPM (b = GRU right / correlated MWPM wrong):

| d | b | c | n_disc | p |
|---|---|---|---|---|
| 5 | 4,218 | 2,794 | 7,012 | 3.4e-65 (GRU better) |
| 7 | 2,060 | 2,648 | 4,708 | 1.1e-17 (correlated MWPM better) |
| 9 | 804 | 1,743 | 2,547 | 6.7e-79 (correlated MWPM better) |

Also in `results/compare_d*_n1800000.json`: Wilson 95% intervals for every pL, McNemar GRU vs MWPM and
correlated vs uncorrelated MWPM, cost, versions. Per-shot decisions of all three decoders:
`results/per_shot_d*_n1800000.npz`.

## Cost (grace1 Grace CPU, one thread per decoder, 1.8M shots)

| d | MWPM decode | correlated decode | correlated / plain | peak RSS of decoder process | whole run (wall, max RSS) |
|---|---|---|---|---|---|
| 5 | 2.3 s (1.3 us/shot) | 6.1 s (3.4 us/shot) | 2.6x | 284 / 277 MiB | 10.5 s, 459 MiB |
| 7 | 7.7 s (4.3 us/shot) | 22.6 s (12.6 us/shot) | 2.9x | 655 / 648 MiB | 33 s, 1.15 GiB |
| 9 | 18.3 s (10.2 us/shot) | 59.0 s (32.8 us/shot) | 3.2x | 1,315 / 1,326 MiB | 81 s, 2.45 GiB |

Matching-graph construction takes about 0.01 s. Decoder peak RSS is almost all the loaded shot arrays (a
child process that only holds them peaks at 275 / 646 / 1,306 MiB); the matcher itself adds at most
about 20 MiB. Pilot first: 10,000 shots at d = 5 (`logs/pilot_d5_n10000.*`).

## Reading

- Against standard MWPM the GRU's advantage is steady with distance (35.0%, 36.8%, 37.7%).
- Correlated matching gains much more with distance (24.5%, 44.5%, 60.7% below standard MWPM): its pL
  falls about 6.4x from d = 5 to d = 9, standard MWPM's 3.3x, the GRU's 3.5x.
- So the GRU beats correlated MWPM only at d = 5 (13.8% lower pL, p = 3e-65). At d = 7 correlated MWPM is
  13.7% better and at d = 9 58.5% better, both significant. The crossover is between d = 5 and d = 7.
- Every "x MWPM" ratio reported elsewhere in this repo is against uncorrelated MWPM.

## Files

`compare_mwpm.py` (the comparison), `run_all.sh` (d = 5, 7, 9 on the full block), `results/`, `logs/`.
Run on grace1 with `~/tfenv/bin/python` from `~/FNAL-AI4QC-GRU-qat/experiments/mwpm_correlated_2026-10-29`.
