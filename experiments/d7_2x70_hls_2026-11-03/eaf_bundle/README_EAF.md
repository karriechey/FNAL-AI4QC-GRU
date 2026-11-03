<!-- Created: 2026-11-03 | Last updated: 2026-11-03 -->
# kd7_2x70 (d7, 2 x GRU 70): Vivado HLS 2020.1 C synthesis

Model kd7_2x70 seed0 epoch 78 (d=7, p=0.004, 54,671 parameters, FP32 pL 0.005224 on the eval block). hls4ml
main 1259e2b25, Vivado backend, io_parallel, Resource strategy, static GRU, ReuseFactor 70, part
xcvu13p-flga2577-2-e, 5 ns clock: the settings of the d7 4 x 70 synthesis. All types ap_fixed<W,I,AP_RND,AP_SAT>,
I from profiling this model.

| project | weights, biases | accumulators, state, input | sigmoid/tanh tables |
|---|---|---|---|
| P8_t4096 | 8 bits | 16 bits | 4096 entries, 16 bits (I=1) |
| P16_t4096 | 16 bits | 16 bits | 4096 entries, 16 bits (I=1) |

P8_t4096 is the counterpart of the 4 x 70 P8_t4096 (514% LUT). P16_t4096 is the 16-bit reference.

Each project has tb_data/ with 100 eval-block shots (inputs, expected hls4ml C-sim logits from grace1, FP32
Keras logits for reference). run_all.sh checks the part, then per project runs C-sim, compares (100/100
decisions, |diff| <= 1e-4) and runs C synthesis only if the C-sim matches.

Expected cost: the 4 x 70 synthesis took about 66 h in total (not timed by its script). This model has one
sequence-output GRU layer instead of three (the layers that held 97% of the 4 x 70 LUT), so expect well under
that per project; run the two projects on two servers. Every step's start, end and hours go to timing.csv
and run_all.out as soon as the step ends.

## Jupyter terminal cells

Upload eaf_d7_2x70_synth.tar.gz to the home folder, then run from scratch:

    mkdir -p /scratch/fast/7DayLifetime/kchey/d7_2x70 && cd /scratch/fast/7DayLifetime/kchey/d7_2x70
    tar xzf ~/eaf_d7_2x70_synth.tar.gz && cd eaf_d7_2x70_synth
    sha256sum -c MANIFEST.sha256 | grep -v ': OK$' ; echo manifest checked
    nohup bash run_all.sh > run_all.out 2>&1 < /dev/null & disown

Both projects on one server run one after another. On two servers, one folder each:

    nohup bash run_all.sh P8_t4096 > run_all.out 2>&1 < /dev/null & disown     # server A
    nohup bash run_all.sh P16_t4096 > run_all.out 2>&1 < /dev/null & disown    # server B

If vivado_hls is not found:

    export VIVADO_SETTINGS=/path/to/Vivado/2020.1/settings64.sh

Progress and completion:

    tail -5 run_all.out
    pgrep -f vivado_hls > /dev/null && echo running || echo finished

Result: results_d7_2x70_<timestamp>.tar.gz in the eaf_d7_2x70_synth folder. Copy it to the home folder to
download:

    cp results_d7_2x70_*.tar.gz ~/
