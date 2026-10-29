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
