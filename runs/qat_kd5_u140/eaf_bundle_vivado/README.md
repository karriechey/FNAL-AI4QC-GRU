<!-- Created: 2026-10-28 | Last updated: 2026-10-28 -->
# kd5_u140 QAT configs: Vivado HLS 2020.1 C synthesis bundle

Four quantized GRU decoders (d=5 student kd5_u140: GRU 140, reset_after=False, Dense 1), as hls4ml
Vivado-backend projects (hls4ml main 1259e2b25), weights already on their ap_fixed grid.

| folder | weights | h and tables | accumulators | eval pL / FP32 |
|---|---|---|---|---|
| qat_w12_s6_seed0_vivado (candidate) | 12 bits | 6 bits | 24 bits | 1.040 |
| qat_w10_s8_seed0_vivado | 10 | 8 | 24 | 1.032 |
| qat_w12_s8_seed0_vivado | 12 | 8 | 24 | 1.004 |
| qat_w16_s6_seed0_vivado | 16 | 6 | 24 | 1.029 |

Target: xcvu13p-flga2577-2-e, 5 ns, io_parallel, Resource strategy, ReuseFactor 70, static GRU.

Same design as the C-sim-verified projects: firmware sources and tcl equal the Vitis-backend projects
(only unused conv / pooling / streaming headers differ), and C-sim logits equal the verified Vitis C-sim
on the first 100,000 eval-block shots (max |diff| 0.0). Details in each `target.json` (`csim_check`).

Run (survives closing the terminal; sources /opt/xilinx/Vivado/2020.1/settings64.sh if needed,
or set VIVADO_SETTINGS):

    cd ~/qat_kd5_u140_synth_vivado
    nohup setsid bash run_all.sh > run_all.log 2>&1 < /dev/null &
    cat run_all.log            # start / done / FAILED per config

A rerun skips configs that already have `synth_results.csv`. Output: `synth_results_vivado.tar.gz`
(synth_all.csv with LUT, FF, DSP, BRAM_18K, URAM, latency, II, estimated clock; per-config csynth logs).
Needs only Vivado HLS and python3 (standard library).
