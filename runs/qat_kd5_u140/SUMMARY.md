<!-- Created: 2026-10-28 | Last updated: 2026-10-28 -->
# kd5_u140 quantization study: summary

Status 2026-10-29 01:10Z. The FP32 reference matched the recorded numbers exactly, so nothing is flagged here.

## What finished, what did not

- Finished: Phase 0 (recon, FP32 reproduced exactly, QGRU check, wiring check), Phase 1 (profiling, both PTQ axes, knees), Phase 2 (7 single-axis QAT points, 6-point 2D grid, 3 seeds for the candidate, its neighbor and the other 18-bit point, plus one diagnostic run).
- Candidate under the pre-registered rule: QAT, weights 12 bits, state 6 bits (18 total, accumulators 24), eval-block LER 0.005353 = 1.040x FP32, 0.707x MWPM (seed 0); 3-seed mean 1.035x FP32, all three seeds acceptable.
- Finished: Phase 3 hls4ml C-sim on the first 1,000,000 eval-block shots for w12_s6, w10_s8, w12_s8, w16_s6 (grace1, aarch64, compile worked without changes, about 22 min each). hls4ml and the quantized Keras model agree on 100% of decisions, max |logit difference| 0.0, for all four; no widening was needed. Exports with run_synth.sh in `hls_export/<config>/`.
- Not done here: C synthesis (no Vitis or Vivado HLS on grace1). Exports for both backends: `for_eaf/eaf_qat_kd5_u140_synth.tar.gz` (sha256 in `eaf_bundle/BUNDLE_SHA256.txt`); synthesis started on EAF with Vivado HLS 2020.1 (first attempt failed because vivado_hls was not on PATH; rerun after sourcing settings64.sh).
- Dropped: QAT state 10 (above the PTQ state knee). Not blocked: nothing.

## Result table (eval block [15.2M, 17M), 1,800,000 shots; FP32 LER 0.005148, MWPM 0.007571 on the same shots)

b = FP32 right and the configuration wrong, c = the reverse; exact two-sided McNemar. Accumulators 24 bits everywhere. hls4ml agreement: Phase 3 table below. Resources: pending (EAF).

| config | weight bits | state bits | method | eval LER | LER/FP32 | LER/MWPM | b / c / n_disc vs FP32 | acceptable |
|---|---|---|---|---|---|---|---|---|
| ptq_w16_s16 | 16 | 16 | PTQ | 0.005158 | 1.002 | 0.681 | 85 / 66 / 151 | yes |
| ptq_w12_s16 | 12 | 16 | PTQ | 0.005243 | 1.018 | 0.693 | 772 / 601 / 1373 | yes |
| ptq_w10_s16 | 10 | 16 | PTQ | 0.008567 | 1.664 | 1.132 | 8514 / 2359 / 10873 | no |
| ptq_w8_s16 | 8 | 16 | PTQ | 0.050307 | 9.772 | 6.645 | 84873 / 3587 / 88460 | no |
| ptq_w6_s16 | 6 | 16 | PTQ | 0.530087 | 102.96 | 70.02 | 949532 / 4642 / 954174 | no |
| ptq_w5_s16 | 5 | 16 | PTQ | 0.195201 | 37.92 | 25.78 | 346264 / 4169 / 350433 | no |
| ptq_w4_s16 | 4 | 16 | PTQ | 0.288210 | 55.98 | 38.07 | 513804 / 4292 / 518096 | no |
| ptq_w16_s12 | 16 | 12 | PTQ | 0.005157 | 1.002 | 0.681 | 113 / 97 / 210 | yes |
| ptq_w16_s10 | 16 | 10 | PTQ | 0.005156 | 1.002 | 0.681 | 263 / 248 / 511 | yes |
| ptq_w16_s8 | 16 | 8 | PTQ | 0.005286 | 1.027 | 0.698 | 971 / 722 / 1693 | yes |
| ptq_w16_s6 | 16 | 6 | PTQ | 0.012968 | 2.519 | 1.713 | 16100 / 2024 / 18124 | no |
| qat_w8_s16 | 8 | 16 | QAT | 0.005742 | 1.115 | 0.758 | 2369 / 1300 / 3669 | no |
| qat_w6_s16 | 6 | 16 | QAT | 0.008005 | 1.555 | 1.057 | 7156 / 2014 / 9170 | no |
| qat_w5_s16 | 5 | 16 | QAT | 0.009283 | 1.803 | 1.226 | 9608 / 2165 / 11773 | no |
| qat_w4_s16 | 4 | 16 | QAT | 0.018177 | 3.531 | 2.401 | 26249 / 2797 / 29046 | no |
| qat_w3_s16 | 3 | 16 | QAT | 0.050461 | 9.801 | 6.665 | 85136 / 3573 / 88709 | no |
| qat_w16_s8 | 16 | 8 | QAT | 0.005170 | 1.004 | 0.683 | 514 / 474 / 988 | yes |
| qat_w16_s6 | 16 | 6 | QAT | 0.005295 | 1.029 | 0.699 | 1141 / 876 / 2017 | yes |
| qat_w12_s8 | 12 | 8 | QAT | 0.005167 | 1.004 | 0.683 | 562 / 527 / 1089 | yes |
| **qat_w12_s6 (candidate)** | 12 | 6 | QAT | 0.005353 | 1.040 | 0.707 | 1310 / 941 / 2251 | yes |
| qat_w12_s6 seed 1 / 2 | 12 | 6 | QAT | 0.005329 / 0.005304 | 1.035 / 1.030 | 0.704 / 0.701 | 1247/921/2168, 1145/864/2009 | yes / yes |
| qat_w12_s5 seeds 0 / 1 / 2 | 12 | 5 | QAT | 0.005531 / 0.005571 / 0.005548 | 1.074 / 1.082 / 1.078 | 0.731-0.736 | 1833/1143, 1988/1226, 1948/1228 | no |
| qat_w10_s8 seeds 0 / 1 / 2 | 10 | 8 | QAT | 0.005315 / 0.005283 / 0.005343 | 1.032 / 1.026 / 1.038 | 0.698-0.706 | 1087/786, 1057/813, 1163/811 | yes |
| qat_w10_s6 | 10 | 6 | QAT | 0.005513 | 1.071 | 0.728 | 1801 / 1144 / 2945 | no |
| qat_w10_s5 | 10 | 5 | QAT | 0.005685 | 1.104 | 0.751 | 2288 / 1321 / 3609 | no |
| qat_w8_s16, LR 3e-5 (diagnostic) | 8 | 16 | QAT | 0.005518 | 1.072 | 0.729 | 1796 / 1129 / 2925 | not used |

All rows (also PTQ at the QAT integer bits, every run's epoch-0 score): `results/all_points.csv`; seed means `results/qat_seeds.csv`.
Plots: `figures/ler_vs_weights_bits.png`, `figures/ler_vs_state_bits.png`, `figures/pareto_ler_vs_total_bits.png` (LER vs total bits; no synthesis numbers yet).

Profiled integer bits (Phase 1, max + 1 headroom): GRU kernel 7 (max 27.8), recurrent kernel 8 (51.9), bias 5 (7.4), Dense kernel 5 (4.6), Dense bias 2 (0.61), GRU accumulator 9 (73.0), h and r*h 3 (1.0), tables 2, Dense accumulator 7. PTQ knees: weights 12, state 8.

## Phase 3: hls4ml C-sim (Vitis backend, io_parallel, Resource, RF 70; first 1,000,000 eval-block shots)

MWPM and FP32 on the same 1M shots. b / c vs FP32 as above.

| config | agreement vs quantized Keras | max abs logit diff | hls4ml LER | LER/FP32 | LER/MWPM | b / c / n_disc vs FP32 | resources (EAF csynth) |
|---|---|---|---|---|---|---|---|
| qat_w12_s6 (candidate) | 100.000% | 0.0 | 0.005304 | 1.039 | 0.704 | 726 / 529 / 1255 | pending |
| qat_w10_s8 | 100.000% | 0.0 | 0.005242 | 1.026 | 0.695 | 588 / 453 / 1041 | pending |
| qat_w12_s8 | 100.000% | 0.0 | 0.005099 | 0.998 | 0.676 | 286 / 294 / 580 | pending |
| qat_w16_s6 | 100.000% | 0.0 | 0.005245 | 1.027 | 0.696 | 631 / 493 / 1124 | pending |

## Open questions

- Weights are the bottleneck: no QAT point at or below 8 weight bits is acceptable. w8 at the planned LR peaked at epoch 1 and its loss rose after; at 1/10 that LR it reached 1.072x. A longer, lower-LR schedule, per-channel integer bits or clipping the weight outliers (recurrent max 52 vs 99th percentile 3.8) are the untested options.
- The early-stop rule (5 evals) ended most runs at epochs 6-20 of 25, before the cosine LR had decayed much.
- w10_s8 has the same total bits as the candidate and a slightly lower mean LER (1.032x vs 1.035x); it loses only on the state-bit tie-break. Synthesis will show which is cheaper.
- All selection used the eval block, so eval-block LERs of the selected points are slightly optimistic; the held-out blocks were not read.
- Accumulators were held at 24 bits throughout; how far they can shrink is not measured.

## Decisions made unattended

1. Output location: code and results in `runs/qat_kd5_u140/` as asked (the repo had no `runs/` folder; other studies live in `experiments/`). On grace1 the working copy is `~/FNAL-AI4QC-GRU-qat` (code and small results); scratch (eval-block arrays, logits, hls4ml projects) in `/tmp/kchey_qat`, because grace1 `/home` had 2.4-2.8 GB free (100% used). `/tmp` is wiped on reboot; everything kept is copied to the repo.
2. hls4ml: upstream main 1259e2b25 (today's HEAD, includes #1568 reset_after=False, #1577 recurrent RF fix, #1589 new Resource cell), as a git worktree at `/tmp/kchey_qat/hls4ml_main`, run with the existing `~/hls4ml_main_env` (TF 2.15.1, tf_keras 2.15.1, qkeras 0.9.0) with the worktree first on PYTHONPATH (`hpy.sh`). A `hls4ml/_version.py` (gitignored, normally made by the installer) was written by hand. The editable install in `~/hls4ml_src` (fix-recurrent-rf-dispatch branch) was not touched.
3. QGRU path skipped (Phase 0.3 default). QKeras QGRU converts with reset_after=False and the same GRU types, but the Dense head takes QKeras types (no AP_RND/AP_SAT) instead of the configured ones, and the GRU weight data written by hls4ml is the same as for the plain Keras GRU (QGRU quantizers are not applied to the data, upstream issue #1579). QAT uses a fake-quantized TF model (`fq_gru.py`) that follows the hls4ml kernel; PTQ uses the same model with the FP32 weights.
4. Precision groups (NOTES.md): weights = GRU kernel, recurrent kernel, bias (and the all-zero recurrent bias), Dense kernel and bias; state = GRU result_t (h and r*h) and table_t (sigmoid and tanh table entries, table_size 1024); accumulators (GRU accum_t, Dense accum/result) held at 24 bits on both axes. Reason: hls4ml stores gate and candidate outputs in accum_t, so a narrow accum_t is also a narrow gate. Input ap_uint<1>. All types AP_RND, AP_SAT (hls4ml defaults truncate and wrap, which needed 24 bits in the u70 study).
5. The GRU table precision is a layer attribute `table_t`, not a `Precision` key; the first wiring run used the hls4ml default ap_fixed<18,8> for the tables. Fixed in `qcommon.hls_config`; the low-precision emulator check (item 9) ran after the fix.
6. Integer bits, Phase 1: I = floor(log2 max|v|) + 2 (sign included) + 1 headroom over all 1.8M eval-block shots. Phase 2 (QAT) uses different integer bits, decided after seeing the Phase 1 profile: weight groups take the I that minimizes squared quantization error of the FP32 weights (outliers: recurrent kernel max 51.9, 99th percentile 3.8), h takes I = 2 and table entries I = 1 (smallest that hold their range). Every QAT run's epoch-0 score is PTQ at its own quantizers, reported as "PTQ (QAT integer bits)".
7. Total bits for the selection rule = weight bits + state bits (accumulators fixed at 24 for all points).
8. QAT recipe details not given in the task: 20M fresh shots per epoch, at most 25 epochs, cosine from 3e-4 to 3e-6 over 25 epochs, TF32 off in training (quantizer rounding), eval block after every epoch. Training shots: the parent's registered seed base 3e12, epochs 80-104 (seed 0), shots the parent never saw; seed s uses epochs [80 + 25 s, 105 + 25 s) (registered max 160). Init is the parent and steps are deterministic, so different shots are the only source of seed variation.
9. Wiring and emulator checks before any sweep: ap_fixed<32,10> everywhere, emulator vs FP32 99.9917% on 1.8M shots (b=85, c=64); hls4ml C-sim vs emulator 99.975% on 20k shots. At low precision (w16_s6 and w5_s16 specs, 5k shots each) C-sim and emulator gave identical logits (max |diff| 0.0).
10. QAT points: state 10 dropped (above the PTQ state knee of 8). Weight points {8, 6, 5, 4, 3} all ran (all below the weight knee of 12).
11. 2D grid (6 points): weights {12, 10} x state {8, 6, 5}. Reason: no QAT weight point at or below 8 bits was acceptable, the smallest acceptable weight width was the PTQ knee 12, and the smallest acceptable state width under QAT was 6.
12. Extra run outside the plan, labelled diagnostic and excluded from selection: w8_s16 at peak LR 3e-5. Reason: w8_s16 at 3e-4 had its best eval at epoch 1 and its training loss rose afterwards.
13. Seeds: the candidate is w12_s6; its neighbor one bit lower is w12_s5 (state one bit lower, same weights). w10_s8, the other acceptable 18-bit point (which wins on LER but loses the state-bit tie-break), also got seeds 1 and 2, in case the candidate fails on its seed mean.
14. Phase 3 configurations: candidate w12_s6 plus w10_s8, w12_s8 (the Pareto front in total bits vs LER) and w16_s6 (smallest acceptable single-axis point). Vitis backend (the export targets Vitis HLS; its GRU kernel is the Vivado one), io_parallel, Resource strategy, ReuseFactor 70, static GRU, xcvu13p-flga2577-2-e, 5 ns. Weights put on their ap_fixed grid in Keras before conversion. C-sim on the first 1,000,000 eval-block shots (C-sim costs about 0.34 core-seconds per shot).
15. A first QAT launch was stopped: `train_qat.py` had no `__main__` guard, so each spawned fresh-shot worker re-ran the script (duplicate evaluations, GPU out of memory with the llama-server sharing the card). Those run folders were deleted and the seven points rerun from scratch.
16. GPU shared with a llama-server (57.5 GB); QAT ran 6-7 jobs at once with a 4 GB cap each.
17. The Mac lost its network for about 5 hours (13:08Z-18:08Z, probably sleep): the GPU was idle in that time. `caffeinate` now keeps the Mac awake.
18. Candidate check with seeds: candidate and neighbor taken from the seed-0 sweep; the seed mean was checked against the same rule afterwards (w12_s6 passes on every seed, so no fallback was needed).
19. Phase 3 order on grace1: the w16_s6 C-sim started first (while the grid trained) was stopped after 26 min and restarted in priority order (candidate first) because the CPU was oversubscribed.
