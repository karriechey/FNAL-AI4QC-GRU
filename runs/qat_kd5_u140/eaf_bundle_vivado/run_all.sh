#!/usr/bin/env bash
# Created: 2026-10-28 | Last updated: 2026-10-28
# Vivado HLS 2020.1 C synthesis of the four kd5_u140 QAT configs, one after another. A config that
# already has synth_results.csv is skipped, so a rerun after a killed session resumes where it stopped.
#   nohup setsid bash run_all.sh > run_all.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
if ! command -v vivado_hls > /dev/null; then
  source "${VIVADO_SETTINGS:-/opt/xilinx/Vivado/2020.1/settings64.sh}" > /dev/null 2>&1
fi
command -v vivado_hls > /dev/null || { echo "vivado_hls not on PATH (set VIVADO_SETTINGS to settings64.sh)"; exit 1; }
echo "$(date -u +%FT%TZ) $(vivado_hls -version 2>&1 | grep -m1 -i vivado)"
for d in qat_w12_s6_seed0_vivado qat_w10_s8_seed0_vivado qat_w12_s8_seed0_vivado qat_w16_s6_seed0_vivado; do
  if [ -f "$d/synth_results.csv" ]; then echo "$(date -u +%FT%TZ) skip $d (done earlier)"; continue; fi
  echo "$(date -u +%FT%TZ) start $d"
  if ( cd "$d" && bash run_synth.sh > run_synth.out 2>&1 ); then
    echo "$(date -u +%FT%TZ) done $d"
  else
    echo "$(date -u +%FT%TZ) FAILED $d (see $d/run_synth.out and $d/synth.log)"
  fi
done
rm -f synth_all.csv
for d in qat_*_vivado; do
  [ -f "$d/synth_results.csv" ] || continue
  if [ -f synth_all.csv ]; then tail -n +2 "$d/synth_results.csv" >> synth_all.csv; else cp "$d/synth_results.csv" synth_all.csv; fi
done
files=$(ls synth_all.csv */synth_results.csv */synth.log */run_synth.out */hls_tool_version.txt 2>/dev/null)
tar czf synth_results_vivado.tar.gz $files run_all.log 2>/dev/null || tar czf synth_results_vivado.tar.gz $files
echo "$(date -u +%FT%TZ) finished; results: $HERE/synth_results_vivado.tar.gz"
