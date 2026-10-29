<!-- Created: 2026-10-29 | Last updated: 2026-10-29 -->
# hls4ml gate for the stacked GRU depth_d7_L04_u70 (CPU only)

Model: d=7, p=0.004, 4 x GRU 70 (reset_after=False) + Dense 1, 113,891 parameters, selected epoch 78 of
`depth_scaling_d7_2026-10/depth_sweep/depth_d7_L04_u70_20261021T170721Z` (grace1). hls4ml main 1259e2b25
(contains 5376da23e), Vivado backend, Resource, io_parallel, static GRU, RF 70, xczu49dr-ffvf1760-2-e, 5 ns.

Inputs: the run's own eval-block per-shot dump [15.2M, 17M) (no pool read), sequences (8, 48) built with the
training layout. FP32 Keras on the CPU gives 6,545 errors (pL 0.003636), equal to the logged run, decisions
100% equal (`results/prep_data.json`).

## Step A: ap_fixed<32,16>, shots [15,200,000, 15,210,000)

Conversion and Python C-sim work for the stacked layers: four GRU layers (return_sequences True, True, True,
False), reset gate before the candidate, one unit in parallel at RF 70. Tables at the hls4ml default
(1024 entries, ap_fixed<18,8>), as in the u70 baseline.

| | value |
|---|---|
| decisions equal to Keras | 9,995 / 10,000 |
| flips with abs Keras logit < 0.2 | 4 |
| flips with abs Keras logit >= 0.2 | 1 (shot 15,200,964: Keras +0.354, hls4ml -0.001, truth 0) |
| errors Keras / hls4ml | 40 / 37 (b = 1, c = 4) |
| logit abs difference, max / mean | 2.90 / 0.130 |

Result: FAIL under the gate rule (one flip at abs logit 0.354). Step B not run. All flips: `results/step_a.json`.

Diagnosis (`step_a_diag.py`, first 2,000 shots, per-layer trace vs Keras sub-models):

| layer | mean abs error, Step A tables | mean abs error, tables 4096 x ap_fixed<32,16> |
|---|---|---|
| gru (h sequence) | 0.0035 | 0.0009 |
| gru_1 | 0.0062 | 0.0018 |
| gru_2 | 0.0096 | 0.0025 |
| gru_3 (last h) | 0.0086 | 0.0022 |
| logit | 0.130 | 0.028 |
| decision flips / flips at abs logit >= 0.2 | 1 / 1 | 0 / 0 |

The error grows layer by layer with no jump at any layer, and 4x finer tables cut it about 4x at every layer.
The divergence is the sigmoid / tanh lookup tables compounding through four layers, not a stacking error in
hls4ml.

## Table ablation (first 2,000 shots, all other types ap_fixed<32,16>)

`table_ablation.py`. 12 tables (sigmoid and tanh for each of the 4 layers, plus the recurrent copies).

| table_size | table_t | table bits | BRAM18K if in BRAM | logit abs err mean / max | flips |
|---|---|---|---|---|---|
| 1024 | ap_fixed<18,8> (default) | 221,184 | 12 | 0.130 / 2.90 | 1 |
| 1024 | ap_fixed<32,16> | 393,216 | 24 | 0.126 / 2.75 | 1 |
| 1024 | ap_fixed<18,1> | 221,184 | 12 | 0.126 / 2.75 | 1 |
| 4096 | ap_fixed<18,8> | 884,736 | 48 | 0.044 / 0.43 | 0 |
| 4096 | ap_fixed<32,16> | 1,572,864 | 96 | 0.028 / 0.28 | 0 |
| 4096 | ap_fixed<18,1> | 884,736 | 48 | 0.028 / 0.28 | 0 |

Table size gives most of the gain (3x lower mean error at 4096). Precision matters only through fractional
bits: ap_fixed<18,1> equals ap_fixed<32,16>, since table values lie in (-1, 1) and the default's 8 integer
bits are unused. Chosen: 4096 x ap_fixed<18,1>, 4x the default table storage.

## Step A again with 4096 x ap_fixed<18,1> tables, same 10,000 shots

| | value |
|---|---|
| decisions equal to Keras | 9,998 / 10,000 |
| flips with abs Keras logit >= 0.2 | 0 (the two flips are at 0.024 and 0.064) |
| errors Keras / hls4ml | 40 / 40 (b = 1, c = 1) |
| logit abs difference, max / mean | 1.40 / 0.029 |

Result: PASS (`results/step_a_t4096_18_1.json`).

## Step B: profiled integer bits, full eval block (1.8M shots)

`step_b.py --table-size 4096`. I = floor(log2 max abs) + 2 from a float64 replay over all shots
(`results/step_b.json`): accumulators 6-7, state 1-2, weights 3-6, biases 1-4, tables 1, input 2. All types
AP_RND, AP_SAT. P16: every type 16 bits. P8: weights and biases 8 bits, everything else 16 bits. Tables
16 bits at 4096 entries, or the hls4ml default as baseline. FP32 Keras: 6,545 errors. Uncorrelated MWPM on
the same shots: pL 0.004291.

| config | errors | pL | x FP32 | x MWPM | McNemar b / c (p) | logit abs diff mean / max |
|---|---|---|---|---|---|---|
| P16_t4096 | 6,582 | 0.003657 | 1.006 | 0.852 | 199 / 162 (0.058) | 0.066 / 5.30 |
| P16_tdefault | 6,593 | 0.003663 | 1.007 | 0.854 | 378 / 330 (0.077) | 0.128 / 6.19 |
| P8_t4096 | 331,522 | 0.1842 | 50.7 | 42.9 | 327,691 / 2,714 (0) | 6.88 / 27.7 |
| P8_tdefault | 380,604 | 0.2114 | 58.2 | 49.3 | 376,811 / 2,752 (0) | |

b = Keras right and hls4ml wrong. At 16 bits both table settings stay within statistics of FP32; the larger
tables halve the logit error and the number of discordant shots (361 vs 708) but move pL by only 11 errors.
8-bit weights with these integer bits keep 2-4 fractional bits, which breaks the model; 8 bits would need
quantization-aware training.

## FPGA cost of the larger tables

Table storage, analytic (12 tables): default 1024 x 18 bits = 221,184 bits (12 BRAM18K if in BRAM); Step B
4096 x 16 bits = 786,432 bits (48 BRAM18K, 4 per table). xczu49dr has 2,160 BRAM18K, so the extra 36 is
1.7% of the device. Measured LUT / FF / BRAM / latency come from the EAF synthesis of the four projects
(t4096 vs tdefault at the same precision).

## EAF bundle

`eaf_bundle/`: the four projects, `run_all.sh` (part check, then per project 100-shot C-sim, compare, C
synthesis; results into results_d7_4x70_<timestamp>.tar.gz), `compare_csim.py`, `collect.py`,
`README_EAF.md`, `MANIFEST.sha256` in the packed copy. Packed as for_eaf/eaf_d7_4x70_synth.tar.gz.
