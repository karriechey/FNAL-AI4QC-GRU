<!-- Created: 2026-11-03 | Last updated: 2026-11-03 -->
# d7 4 x GRU 70 on larger parts: EAF steps

Needs python3 (standard library), bash and the Xilinx tools. No pip, no network. Upload
`eaf_d7_4x70_large_parts.tar.gz` to the home folder, then work from scratch (home is small, scratch deletes after 7 days).

## 1. Unpack and check the tools (minutes)

    mkdir -p /scratch/fast/7DayLifetime/kchey/d7_parts && cd /scratch/fast/7DayLifetime/kchey/d7_parts
    tar xzf ~/eaf_d7_4x70_large_parts.tar.gz && cd eaf_d7_4x70_large_parts
    sha256sum -c MANIFEST.sha256 | grep -v ': OK$' ; echo manifest checked
    python3 preflight.py

The report prints at the end and is saved in `preflight/`. For a part that fails, the line starts with the reason:
TOOL, PART, LICENSE or OTHER. If the tools are in a folder the script does not search:

    export XILINX_ROOTS=/path/to/xilinx

Bring back `~/d7_4x70_large_parts_results/preflight_report_*.txt` if any part fails.

## 2. Stage A: one run, RF 70 on the VU13P

    nohup python3 run_synth.py --stage A > run_A.out 2>&1 < /dev/null & disown

The earlier run of this design took 66 h and gave 8.88M LUT and no GRU latency. This run has the output loop fix.
Look at its LUT and hours before starting stage B (`tail run_A.out`). Stage B is worth running if LUT is far below 8.88M.

## 3. Stages B and C: one server per list

Run preflight on each new server first (it writes `preflight/selected.tsv` in that server's folder). Then:

    nohup python3 run_synth.py vu19p_P8_t4096_rf70 vu19p_P8_t4096_rf35 > run_vu19p.out 2>&1 < /dev/null & disown
    nohup python3 run_synth.py vp1902_P8_t4096_rf70 vp1902_P8_t4096_rf35 vp1902_P8_t4096_rf20 > run_vp1902.out 2>&1 < /dev/null & disown
    nohup python3 run_synth.py vu13p_P8_t4096_rf14 > run_vu13p.out 2>&1 < /dev/null & disown

Stage C is the working 16-bit decoder at the lowest RF each part fits:

    nohup python3 run_synth.py --stage C > run_C.out 2>&1 < /dev/null & disown

Jobs that preflight did not pass are skipped with a message. A job stops after 144 h (`export JOB_TIMEOUT_H=...`).
A finished job is skipped on a rerun.

## 4. Bring back

Every job adds its row to one CSV in the home folder, which all servers share:

    ~/d7_4x70_large_parts_results/large_parts_results.csv
    ~/d7_4x70_large_parts_results/reports_*.tar.gz

Progress:

    tail -3 run_A.out
    pgrep -af "vivado_hls|vitis" | cut -c1-80
