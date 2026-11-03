<!-- Created: 2026-11-03 | Last updated: 2026-11-03 -->
# hls4ml PTQ for kd7_2x70 (d7, 2 x GRU 70), the 2-layer counterpart of d7_4x70_hls_2026-10-29

Question: is the 514% LUT of the 4 x 70 P8_t4096 synthesis set by stacking depth or by model size?

Model: d=7, p=0.004, 2 x GRU 70 (reset_after=False) + Dense 1, 54,671 parameters, selected epoch 78 of
`kd_pilot_p004_2026-10/runs/kd7_2x70_20261019T011509Z` (grace1; distilled from f7w_p004). FP32 pL 0.005224 on
the eval block, 1.218 x uncorrelated MWPM. Same settings as the 4 x 70: hls4ml main 1259e2b25 (contains
5376da23e), Vivado backend, Resource, io_parallel, static GRU, RF 70, 5 ns, xcvu13p-flga2577-2-e, input
(8, 48), 4096-entry tables at ap_fixed<16,1>, P8 and P16 as defined below. Post-training quantization only.

Files on grace1 live in /tmp (/home is full): code in /tmp/kchey_qat/code/d7_2x70_hls_2026-11-03, projects in
/tmp/kchey_qat/d7_2x70.

## Parameters and layer outputs

GRU parameters 3 (u (n_in + u) + u), reset_after=False.

| layer | 4 x 70 | 2 x 70 | output |
|---|---|---|---|
| gru (48 -> 70) | 24,990 | 24,990 | full sequence (8 x 70) in both |
| gru_1 (70 -> 70) | 29,610 | 29,610 | full sequence in 4 x 70, last state in 2 x 70 |
| gru_2 (70 -> 70) | 29,610 | | full sequence |
| gru_3 (70 -> 70) | 29,610 | | last state |
| dense | 71 | 71 | |
| total | 113,891 | 54,671 (0.48 x) | |

The 4 x 70 C synthesis report (for_eaf/d7_P8_t4096_csynth.rpt) splits the LUT by layer:

| 4 x 70 instance | output | LUT | DSP |
|---|---|---|---|
| gru | full sequence | 3,274,450 | 390 |
| gru_1 | full sequence | 2,697,300 | 290 |
| gru_2 | full sequence | 2,673,388 | 503 |
| gru_3 | last state | 163,473 | 427 |
| dense | | 3,086 | |
| total | | 8,883,465 (514%) | 1,611 |

The three full-sequence layers hold 97% of the LUT; gru_3 has the same parameter count as gru_1 and gru_2 and
16x fewer LUT. If the per-layer cost carries over, 2 x 70 lands near 3.4M LUT (about 200%): one full-sequence
layer plus one last-state layer. A result near half of 8.9M would point to parameter count instead.

## Agreement with Keras

Prep (`results/prep_data.json`): FP32 Keras on the CPU gives 9,404 errors on the eval block [15.2M, 17M),
equal to the logged run, decisions 100% equal.

Step A (`results/step_a_t4096_18_1.json`, shots [15,200,000, 15,210,000), ap_fixed<32,16>, tables 4096 x
ap_fixed<18,1>): 9,999 / 10,000 decisions equal; one flip at Keras logit 0.040; no flip with abs Keras logit
>= 0.2; logit abs difference max / mean 0.259 / 0.035. PASS.

Step B (`results/step_b.json`, 1.8M shots): I = floor(log2 max abs) + 2 from a float64 replay. Integer bits:
accumulators 6-8, state 1-2, weights 4-6, biases 1-4, tables 1, input 2. All types AP_RND, AP_SAT. P16: every
type 16 bits. P8: weights and biases 8 bits, everything else 16 bits. tdefault: hls4ml default tables (1024 x
ap_fixed<18,8>), for the table-size cost comparison (`results/step_b_P8_tdefault.json`).

| config | errors | pL | x FP32 | x MWPM | McNemar b / c (p) | logit abs diff mean / max |
|---|---|---|---|---|---|---|
| P16_t4096 | 9,467 | 0.005259 | 1.007 | 1.226 | 474 / 411 (0.037) | 0.125 / 2.32 |
| P8_t4096 | 493,441 | 0.2741 | 52.5 | 63.9 | 488,322 / 4,285 (0) | 8.32 / 23.4 |
| P8_tdefault | 494,857 | 0.2749 | 52.6 | 64.1 | 489,743 / 4,290 (0) | / 23.4 |
| 4 x 70 P16_t4096 | 6,582 | 0.003657 | 1.006 | 0.852 | 199 / 162 (0.058) | 0.066 / 5.30 |
| 4 x 70 P8_t4096 | 331,522 | 0.1842 | 50.7 | 42.9 | 327,691 / 2,714 (0) | 6.88 / 27.7 |

b = Keras right and hls4ml wrong. P16 stays within 0.7% of FP32. P8 breaks this model the same way it breaks
the 4 x 70: the first layer's weights reach abs 21.8 (input) and 26.6 (recurrent), so 6 integer bits leave 2
fractional bits. The P8 projects of both models therefore measure hardware cost at 8-bit weights only; a
working 8-bit decoder needs quantization-aware training.

## EAF bundle

`eaf_bundle/`: run_all.sh (part check, per project 100-shot C-sim against the grace1 hls4ml C-sim, C
synthesis; start, end and hours of every step in timing.csv), compare_csim.py, collect.py, README_EAF.md.
Packed with both projects and MANIFEST.sha256 as for_eaf/eaf_d7_2x70_synth.tar.gz (1.3 MB);
P8_tdefault alone as for_eaf/eaf_d7_2x70_P8_tdefault.tar.gz. The 4 x 70
synthesis took about 66 h.
