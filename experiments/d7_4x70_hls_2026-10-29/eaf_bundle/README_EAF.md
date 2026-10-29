<!-- Created: 2026-10-29 | Last updated: 2026-10-29 -->
# d7 stacked GRU (4 x 70): Vivado HLS 2020.1 C synthesis

Model depth_d7_L04_u70 seed0 epoch 78 (d=7, p=0.004, FP32 pL 0.003636 on the eval block). hls4ml main
1259e2b25, Vivado backend, io_parallel, Resource strategy, static GRU, ReuseFactor 70, part
xczu49dr-ffvf1760-2-e, 5 ns clock. All types ap_fixed<W,I,AP_RND,AP_SAT>, I from profiling.

| project | weights, biases | accumulators, state, input | sigmoid/tanh tables |
|---|---|---|---|
| P16_t4096 | 16 bits | 16 bits | 4096 entries, 16 bits (I=1) |
| P8_t4096 | 8 bits | 16 bits | 4096 entries, 16 bits (I=1) |
| P16_tdefault | 16 bits | 16 bits | hls4ml default, 1024 x ap_fixed<18,8> |
| P8_tdefault | 8 bits | 16 bits | hls4ml default, 1024 x ap_fixed<18,8> |

The tdefault pair is the baseline for the cost of the larger tables.

Each project has tb_data/ with 100 eval-block shots: tb_input_features.dat, tb_output_predictions.dat (the
hls4ml C-sim logits on grace1, which run_all.sh compares against) and keras_fp32_logits_100.dat (reference
only). run_all.sh checks the part, then per project runs C-sim, compares (100/100 decisions, |diff| <= 1e-4),
and runs C synthesis only if the C-sim matches. Needs bash, python3 (standard library) and Vivado HLS.

Expected cost per project (estimate; no synthesis of this design has run yet): 2-5 h wall, 4-10 GB peak.
The C-sim step takes a few minutes.

## Jupyter terminal cells

Upload eaf_d7_4x70_synth.tar.gz to the home folder, then run from scratch (home is shared and small).

All four on one server:

    mkdir -p /scratch/fast/7DayLifetime/kchey/d7_4x70 && cd /scratch/fast/7DayLifetime/kchey/d7_4x70
    tar xzf ~/eaf_d7_4x70_synth.tar.gz && cd eaf_d7_4x70_synth
    sha256sum -c MANIFEST.sha256 | grep -v ': OK$' ; echo manifest checked
    nohup bash run_all.sh > run_all.out 2>&1 < /dev/null & disown

Split across servers (one folder per server, same commands with a project list):

    nohup bash run_all.sh P16_t4096 P16_tdefault > run_all.out 2>&1 < /dev/null & disown    # server A
    nohup bash run_all.sh P8_t4096 P8_tdefault > run_all.out 2>&1 < /dev/null & disown      # server B

If vivado_hls is not found, set the settings path first:

    export VIVADO_SETTINGS=/path/to/Vivado/2020.1/settings64.sh

Progress and completion:

    tail -5 run_all.out
    pgrep -f vivado_hls > /dev/null && echo running || echo finished

Result: results_d7_4x70_<timestamp>.tar.gz in the eaf_d7_4x70_synth folder. Copy it to the home folder to
download:

    cp results_d7_4x70_*.tar.gz ~/
