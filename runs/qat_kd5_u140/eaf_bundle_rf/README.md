<!-- Created: 2026-10-29 | Last updated: 2026-10-29 -->
# kd5_u140 QAT candidate: ReuseFactor sweep, Vivado HLS 2020.1 C synthesis

Candidate qat_w12_s6 (12-bit weights, 6-bit h and tables, 24-bit accumulators) at ReuseFactor 10, 35 and
140; RF 70 is in the earlier bundle. Same RF values as the u70 sweep. hls4ml main 1259e2b25, Vivado
backend, io_parallel, Resource, static GRU, xcvu13p-flga2577-2-e, 5 ns. Each folder's C-sim logits equal
the verified RF 70 C-sim on the first 100,000 eval-block shots (max |diff| 0.0), so pL does not depend on RF.

| folder | RF | GRU units in parallel | expected latency vs RF 70 |
|---|---|---|---|
| qat_w12_s6_seed0_rf10_vivado | 10 | 14 | about 0.15x; largest design |
| qat_w12_s6_seed0_rf35_vivado | 35 | 4 | about 0.5x |
| qat_w12_s6_seed0_rf140_vivado | 140 | 1 | about 2x |

One config per server, each in its own scratch folder (survives closing the terminal):

    mkdir -p /scratch/fast/7DayLifetime/kchey/qat_rf10 && cd /scratch/fast/7DayLifetime/kchey/qat_rf10
    tar xzf ~/eaf_qat_kd5_u140_rf_sweep.tar.gz
    nohup bash run_one.sh qat_w12_s6_seed0_rf10_vivado > run_rf10.log 2>&1 < /dev/null & disown

(rf35 and rf140 the same on the other two servers.) Result per server:
`synth_results_<folder>.tar.gz` next to `run_one.sh`. Needs only Vivado HLS and python3.
