<!-- Created: 2026-11-03 | Last updated: 2026-11-04 -->
# d7 4 x GRU 70 on larger parts: VP1902, VU19P, with VU13P as reference

Status 2026-11-05: stage A done. The output loop patch cuts P8_t4096 RF 70 from 8.88M to 1.44M LUT, which fits the VU13P.
VU19P and VP1902 are blocked by the EAF install.

## Preflight on EAF (jupyter-kchey-seed1, 126 CPUs)

Tools found: Vivado HLS 2020.1 and Vivado 2020.1 in `/opt/xilinx/Vivado/2020.1`. No Vitis HLS.

| part | result | detail |
|---|---|---|
| xcvu13p-flga2577-2-e | pass | C-sim and C synthesis run; Vivado implementation license reported missing |
| xcvu19p (fsvb3824-2-e, fsva3824-2-e) | blocked: part | `[HLS 200-70] Part ... is not installed`; absent from the Vivado part list |
| xcvp1902-vsva6865-2MP-e-S | blocked: tool | `[HLS 200-70] Part ... is not installed`; needs Vitis HLS with Versal Premium device files |

`XILINXD_LICENSE_FILE` and `LM_LICENSE_FILE` are both empty. C synthesis numbers here are HLS estimates, with no Vivado synthesis
or place and route. Runnable jobs: VU13P P8_t4096 RF 70 and 14, VU13P P16_tdefault RF 14.

Model: depth_d7_L04_u70 seed 0 epoch 78, the checkpoint behind P8_t4096. No QAT exists for this model, so these are
post-training quantization configs. hls4ml main 1259e2b25, io_parallel, Resource, 5 ns.

## Table size gate

`table_gate.py`, full eval block [15.2M, 17M), existing hls4ml C-sim logits from Step B of `d7_4x70_hls_2026-10-29`.
b = 4096 right and 1024 wrong, c = the reverse.

| config | pL 4096 | pL 1024 | errors 4096 / 1024 | b / c | McNemar p | result |
|---|---|---|---|---|---|---|
| P8 (8-bit weights) | 0.18418 | 0.21145 | 331,522 / 380,604 | 54,620 / 5,538 | 0 | fail, keep 4096 |
| P16 (16-bit weights) | 0.003657 | 0.003663 | 6,582 / 6,593 | 311 / 300 | 0.69 | pass, use 1024 |

The P8 decoder is broken at either table size (50x FP32), since 8-bit weights leave 2-4 fractional bits. P8_t4096 measures
hardware cost only. P16_tdefault is the working decoder (1.007x FP32, 0.854x uncorrelated MWPM).

## What the earlier synthesis shows

P8_t4096, RF 70, VU13P (Vivado HLS 2020.1): 8,883,465 LUT, 1,611 DSP, 66 h. That LUT count is 514% of VU13P, 217% of VU19P
and 105% of VP1902, so no target part has room at this RF.

The three full-sequence layers hold 97% of the LUT (2.7-3.3M each); the last-state layer has the same parameters and 163k.
They also report latency "?". The log points to one loop in `gru_stack`, which copies the state to the output:

    WARNING: [XFORM 203-503] Cannot unroll loop 'Loop-2.2' (nnet_recurrent.h:1033) ... completely: variable loop bound.

Its bounds depend on `iloop`, so HLS keeps it as a loop and muxes the partitioned `h_state`. The input loop just above it has
constant bounds and unrolls.

## Output loop patch

`make_projects.py` rewrites the loop in each generated project (hls4ml itself is unchanged):

    for (int j = 0; j < CONFIG_T::n_state; j++) { res[j + CONFIG_T::n_state * iloop] = h_state[j]; }

Logits are identical to the stock RF 70 C-sim on the first 100,000 eval-block shots for all 10 projects (max abs diff 0.0,
`results/make_projects_n100000*.json`), so pL carries over. Effect on LUT and runtime: see Results.

## Tools and backends

- VU13P, VU19P: Vivado backend, Vivado HLS 2020.1, the tool behind every earlier number.
- VP1902: Vitis backend, Vitis HLS. hls4ml calls `vitis-run`; the scripts use `vitis_hls` if present, else `vitis-run`.
  Its default clock uncertainty is 27% against 12.5% for Vivado, kept as hls4ml sets it, so compare the two tools at RF 70.
- Part names: VP1902 order code XCVP1902-2MSEVSVA6865 taken as Vivado part xcvp1902-vsva6865-2MP-e-S. `preflight.py` tries
  this, then whatever `get_parts xcvp1902*` lists, and flags any fallback. VU19P tries fsvb3824-2-e, then fsva3824-2-e.

## Reuse factors

Stock RF 70 uses 1,611 DSP, about one per multiply per cycle, so DSP scales near 70 / RF. Valid RFs follow the cell schedule
(`n_unit_par`, `n_cycles`). Cycles per layer are about 8 x (2 x RF + 61) + 17, from stock RF 70 (1,625). The 61 is pipeline
depth and does not shrink with RF, so going below RF 14 gains little. Estimates only:

| RF | n_unit_par | DSP | % VU13P | % VU19P | % VP1902 | latency estimate at 5 ns |
|---|---|---|---|---|---|---|
| 70 | 1 | 1,611 | 13 | 42 | 23 | 6,572 cycles, 32.9 us |
| 35 | 2 | 3,222 | 26 | 84 | 47 | 4,300 cycles, 21.5 us |
| 20 | 7 | 5,639 | 46 | 147 | 82 | 3,340 cycles, 16.7 us |
| 14 | 5 | 8,055 | 66 | 210 | 117 | 2,950 cycles, 14.8 us |

Lowest RF that fits DSP: VU19P 35, VP1902 20, VU13P 14. Dense stays at RF 35 or above, because lower RF chained 4 or more
saturating adds per cycle and missed 5 ns on kd5_u140 (RF 35 there, 8.17 ns). RF 10 failed on kd5_u140 at the unroll limit.

## Jobs (`eaf_bundle/jobs.tsv`)

| stage | jobs | why |
|---|---|---|
| A | VU13P P8_t4096 RF 70 | compares directly with the stock 8.88M LUT, shows whether the patch works |
| B | VU19P P8_t4096 RF 70, 35; VP1902 P8_t4096 RF 70, 35, 20; VU13P P8_t4096 RF 14 | the asked-for parts at their lowest RF |
| C | P16_tdefault at VU13P RF 70, VU19P RF 35, VP1902 RF 20, VU13P RF 14 | the working decoder at the same limits, and at RF 70 to compare with P8 |

VU13P stock RF 70 already exists (sheet tab d7_4x70 HLS, 66.05 h), so it is not rerun.

## Results

`results/large_parts_results.csv`, copied from `~/d7_4x70_large_parts_results/` on EAF. HLS C-synthesis estimates, sheet tab d7_4x70 HLS.

P8_t4096, RF 70, VU13P, Vivado HLS 2020.1, 5 ns, on jupyter-kchey-seed1 (AMD EPYC 7542, 126 CPUs):

| | stock | output loop patch |
|---|---|---|
| LUT | 8,883,465 (514%) | 1,438,201 (83.23%) |
| FF | 454,342 (13.15%) | 229,355 (6.64%) |
| DSP | 1,611 (13.11%) | 1,611 (13.11%) |
| BRAM_18K | 49 (0.91%) | 49 (0.91%) |
| URAM | 0 | 0 |
| latency | unknown | 6,584 cycles, 32.92 us |
| II | not recorded | 1,634 cycles, 8.17 us |
| est clock | 4.374 ns | 4.345 ns |
| wall time, peak memory | 66.05 h, 33.5 GB | 12.82 h, 9.37 GB |

The patched LUT count would be 35.2% of VU19P and 17.0% of VP1902, with DSP at 42.0% and 23.5%. Those parts were not synthesized.

| layer | latency cycles | LUT |
|---|---|---|
| gru | 1,633 | 792,824 |
| gru_1 | 1,625 | 215,481 |
| gru_2 | 1,625 | 191,569 |
| gru_3 | 1,625 | 163,473 |
| logit | 72 | 3,086 |

The first layer has the fewest weights (48 inputs against 70) and 3.7-4.8x the LUT of the others, cause unknown. Latency matches the
schedule estimate of 6,572 cycles. pL is 0.18418 (50.65x FP32), from the stock full-block C-sim.

LUT at lower RF is unmeasured, and RF 70 already uses 83% of the VU13P.
