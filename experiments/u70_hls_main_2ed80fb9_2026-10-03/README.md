<!-- Created: 2026-10-03 | Last updated: 2026-10-15 -->
# u70 GRU on hls4ml main 2ed80fb9: C-sim gate, PTQ width sweep, QGRU check

Model: d=5 r=5 p=0.004 GRU, 70 units, `reset_after=False`, 20,021 parameters (`gkd20m_u70_hard_seed0`,
trained on the EAF x86 FORMAL pool). Data: FORMAL evaluation block [15,200,000, 17,000,000) of that
pool (non-sealed; the sealed block [17M, 19M) is not in `data/` and `u70_common.load_eval` asserts the
shot range). hls4ml: upstream main `2ed80fb9a6d5150a7f1ec21a8cf6564900244a0e`, git install in
`~/hls4ml_main_env` on grace1 (aarch64, g++ 14.3.1); every result JSON records the commit.
Step 2 (Vivado HLS synthesis) ran on EAF from `for_eaf/eaf_u70_main_rf_sweep.tar.gz` on 2026-10-15.

## Step 1: C-sim gate, ap_fixed<32,16> ("upper-bound baseline, <32,16> PTQ")

Resource, static GRU, io_parallel, xcvu13p, 5 ns.

| requested RF | effective RF (input / recurrent / Dense) | first 10,000 shots | all 1,800,000 shots |
|---|---|---|---|
| 35 | 24 / 35 / 35 | decisions 100% equal; errors 66 vs 66; b=0 c=0 | errors HLS 12,638 vs Keras 12,619; b=136 c=117 n_disc=253; p=0.26; decisions 99.986% equal |
| 140 | 144 / 140 / 70 | same as RF 35 (logits bit-identical) | not run |
| 70, stock main | 72 / 70 / 70 | C sim aborts (assert in `dense_resource_rf_gt_nin_rem0`) | n/a |
| 70, kernel fix | 72 / 70 / 70 | same as RF 35 (logits bit-identical) | not run |

Logits are never bit-identical to Keras (max |d| 0.40 on 10k, 0.87 on 1.8M; the 253 flipped shots all
have |Keras logit| <= 0.156): the sigmoid/tanh lookup tables (1024 entries on [-8, 8)) dominate.
The full-block numbers equal the 2026-09-29 PR-head run exactly.

RF 70 failure: `hls4ml/backends/vivado/passes/recurrent_templates.py:243` chooses the recurrent
multiply's kernel from the input multiply's reuse factor (72 > n_in 70) but writes the recurrent
reuse factor (70). `patch_recurrent_kernel_choice.py` is the one-line local fix.

## Step 2: ReuseFactor sweep, Vivado HLS 2020.1 C synthesis on EAF ("upper-bound baseline, <32,16> PTQ")

Run 2026-10-15 (`results/eaf_rf_sweep_20261015T194454Z/`). Same configuration as Step 1. RF 70 uses the
kernel-choice fix (`_kernelfix`); upstream main has the same fix since #1577 (5376da23e, 2026-10-14).
Each run passed the Python C-sim gate (10,000 shots, decisions 100% equal, errors 66 vs 66) and
Vivado C sim (100 shots, 100/100). C synthesis only, no co-simulation.

| requested RF | effective RF (input / recurrent / Dense) | latency cycles | latency at 5 ns | II | DSP | LUT | FF | BRAM18K | est. clock |
|---|---|---|---|---|---|---|---|---|---|
| 10 (2026-09-29, PR head 38affb5) | 8 / 10 / 10 | 247-260 | 1.24-1.30 us | 236-248 | 9,262 (75.4%) | 694,953 (40.2%) | 239,969 (6.9%) | 1,978 (36.8%) | 4.328 ns |
| 35 | 24 / 35 / 35 | 560-573 | 2.80-2.87 us | 524-536 | 3,362 (27.4%) | 540,780 (31.3%) | 119,442 (3.5%) | 667 (12.4%) | 4.328 ns |
| 70 (kernelfix) | 72 / 70 / 70 | 1,027-1,040 | 5.14-5.20 us | 956-968 | 1,958 (15.9%) | 434,343 (25.1%) | 96,256 (2.8%) | 355 (6.6%) | 4.328 ns |
| 140 | 144 / 140 / 70 | 1,879-1,892 | 9.40-9.46 us | 1,808-1,820 | 1,398 (11.4%) | 424,910 (24.6%) | 91,849 (2.7%) | 230 (4.3%) | 4.328 ns |

- Device: xcvu13p-flga2577-2-e. No timing violations. Synthesis took 6-11 min per run (peak 2.4-3.3 GB).
- Latency and II grow almost linearly with RF; DSP and BRAM fall with RF.
- LUT levels off near 425k (about 25%) above RF 70. XFORM 203-104 warnings: `h_state`, `layer2_out` and
  (RF 70, 140) `acc` are completely partitioned and read with non-constant indices, so LUTs go to
  multiplexers that the RF does not reduce. At 32 bits these and the activation tables set the LUT floor.

## Step 3: PTQ width sweep (C simulation only, 1,800,000 shots)

Integer bits from profiling (`results/step3_profiled_ranges.json`, I = floor(log2 max|v|) + 2):
GRU weight 6 (max |w| 25.9), recurrent weight 7 (47.0), biases 5 (8.65), accumulator 8 (81.9),
h 2, tables 1, Dense weight 4, Dense bias 1, Dense accumulator/result 6 (20.8), input 2.
Full table: `results/step3_ptq_table.md` / `.csv`.

| W | trn_wrap (hls4ml default): HLS errors, b / c, p | rnd_sat (round, saturate): HLS errors, b / c, p |
|---|---|---|
| FP32 Keras | 12,619 | 12,619 |
| 24 | 12,643, 137 / 113, p=0.15 | 12,654, 135 / 100, p=0.026 |
| 18 | 12,773, 585 / 431, p=1.5e-6 | 12,618, 119 / 120, p=1.0 |
| 16 | 13,537, 2,258 / 1,340, p=2.5e-53 | 12,641, 205 / 183, p=0.29 |
| 12 | 247,343 | 14,095, 3,457 / 1,981, p=4.2e-90 |
| 10 | 374,293 | 153,531 |
| 8 | 456,150 (accumulator capped) | 469,185 (accumulator capped) |

- Smallest width indistinguishable from FP32: **16 bits with rounding + saturation** (ΔLER 95% CI
  [-9e-6, +3.4e-5]); with the hls4ml default truncation, 24 bits.
- rnd_sat W=24 (p=0.026) does not follow the strict rule "it and every wider width pass". Restoring any
  single type to <32,16> does not change it (`results/step3_ablation_rnd_sat_w24.json`, 27-32 flips
  per 200k for every group), and the <32,16> baseline has the same discordant pattern (136/117), so it
  is the lookup-table effect common to all widths. 12 tests at α=0.05: Bonferroni threshold 0.0042.
- What fails first: no type overflows its profiled range above W=8 (only the accumulator is capped, at
  W=8). The failure is lost fractional bits. Under truncation the GRU accumulator limits first: at W=18,
  restoring it alone cuts decision flips from 97 to 36 per 200k (p 0.15 -> 0.62); restoring any other
  group does nothing. Large-error ("overflow signature", |err| > 0.5) shots appear first in the logit
  (trn_wrap W=18: 2.9% of trace shots; rnd_sat W=12: 53%), then in h (rnd_sat W=12: 6.4%).

## Step 4: QKeras QGRU (8 units, reset_after=False, random weights, 10,000 shots)

| variant | parses | C sim vs Keras |
|---|---|---|
| quantized_tanh(8) / quantized_sigmoid(8) | no: `AssertionError: Hard activation not supported` | n/a |
| use_real_tanh / use_real_sigmoid | yes, generates `reset_after = false`, bias 24 + 24 | max abs diff 0.031, decisions 97.3% equal |
| float activations, quantized weights | yes | max abs diff 0.026, 98.2% |
| same, other types widened to <32,16> | yes | max abs diff 0.022, mean 0.0032, 98.8% |
| same, weights pre-quantized in Keras | yes | max abs diff 0.0049, mean 0.0011, 99.4% |
| control: Keras GRU, same quantized weights | yes | max abs diff 0.0051, mean 0.0010, 99.4% |

Random-weight logits sit near 0 (range -0.28 to 0.79), so decision agreement is a weak measure here.
hls4ml does not apply the QGRU kernel/recurrent quantizers to the weight data (`GRU.initialize` in
`hls4ml/model/layers.py` calls `add_weights` without the quantizer, unlike `Dense`): the raw float
weights are written and cast to ap_fixed<8,1> by truncation. Pre-quantizing the weights brings the
error down to the plain-GRU control.

## Other hls4ml issues met

`hls4ml.model.profiling.weights_keras` raises IndexError on a GRU layer; `get_ymodel_keras` tries to
assign `GRU.activation` (read-only in Keras 2.15). Step 3 uses its own numpy replay and Keras sub-models.

## Files

`u70_common.py` (model rebuild, data, statistics, hls4ml commit check), `step1_csim_gate.py`,
`step3_ptq_sweep.py`, `step4_qgru_check.py`, `patch_recurrent_kernel_choice.py`, `results/`, `logs/`
(`logs/step1_n1800000.log` is the first RF 70 attempt that aborted). `data/` and `hls_*` projects are gitignored.
