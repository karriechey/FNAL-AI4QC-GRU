<!-- Created: 2026-10-28 | Last updated: 2026-10-28 -->
# kd5_u140 quantized GRU: C synthesis bundle

Four QAT configurations of the d=5 distilled GRU student kd5_u140 (GRU 140, reset_after=False,
Dense 1), weights already on their ap_fixed grid. Each folder holds a generated hls4ml project
(hls4ml main 1259e2b25, generated on aarch64), the hls4ml config, precisions, versions and target.
Two copies per configuration: `_vitis` (Vitis backend) and `_vivado` (Vivado backend), same precisions.

| folder | weights | state (h, tables) | accumulators |
|---|---|---|---|
| qat_w12_s6_seed0 (candidate) | 12 bits | 6 bits | 24 bits |
| qat_w10_s8_seed0 | 10 | 8 | 24 |
| qat_w12_s8_seed0 | 12 | 8 | 24 |
| qat_w16_s6_seed0 | 16 | 6 | 24 |

Target: xcvu13p-flga2577-2-e, 5 ns clock, io_parallel, Resource strategy, ReuseFactor 70, static GRU.

    bash run_all.sh vivado     # or: bash run_all.sh vitis

C synthesis only (no C-sim, cosim or export); results in `synth_all.csv` and `synth_results_<kind>.tar.gz`.
`RECONVERT=1 bash <config>/run_synth.sh` regenerates a project from keras_model.json with hls4ml 1259e2b25 installed.
