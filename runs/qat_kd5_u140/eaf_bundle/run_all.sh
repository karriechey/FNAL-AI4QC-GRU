#!/usr/bin/env bash
# Created: 2026-10-28 | Last updated: 2026-10-28
# C synthesis of every exported config, one after another (xcvu13p-flga2577-2-e, 5 ns, Resource, RF 70).
#   bash run_all.sh vivado     Vivado HLS (vivado_hls on PATH, e.g. 2020.1)
#   bash run_all.sh vitis      Vitis HLS (vitis_hls or vitis-run on PATH)
# Results: <config>/synth_results.csv, all rows in synth_all.csv. Needs only python3 (standard library).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
kind=${1:?vivado or vitis}
cd "$HERE"
if [ "$kind" = vivado ] && ! command -v vivado_hls > /dev/null; then
  source "${VIVADO_SETTINGS:-/opt/xilinx/Vivado/2020.1/settings64.sh}" > /dev/null 2>&1
fi
if [ "$kind" = vivado ]; then
  command -v vivado_hls > /dev/null || { echo "vivado_hls not on PATH (set VIVADO_SETTINGS to settings64.sh)"; exit 1; }
fi
rm -f synth_all.csv
for d in qat_w12_s6_seed0_$kind qat_w10_s8_seed0_$kind qat_w12_s8_seed0_$kind qat_w16_s6_seed0_$kind; do
  echo "$(date -u +%FT%TZ) start $d"
  ( cd "$d" && bash run_synth.sh > run_synth.out 2>&1 ) && echo "$(date -u +%FT%TZ) done $d" || echo "$(date -u +%FT%TZ) FAILED $d (see $d/synth.log)"
  if [ -f "$d/synth_results.csv" ]; then
    if [ -f synth_all.csv ]; then tail -n +2 "$d/synth_results.csv" >> synth_all.csv; else cp "$d/synth_results.csv" synth_all.csv; fi
  fi
done
files=$(ls synth_all.csv */synth_results.csv */synth.log */run_synth.out 2>/dev/null)
tar czf synth_results_$kind.tar.gz $files
echo "results: $HERE/synth_results_$kind.tar.gz"
