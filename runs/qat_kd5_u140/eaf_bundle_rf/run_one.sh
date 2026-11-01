#!/usr/bin/env bash
# Created: 2026-10-29 | Last updated: 2026-10-29
# Vivado HLS 2020.1 C synthesis of one config of this bundle; one server per config.
#   nohup bash run_one.sh qat_w12_s6_seed0_rf10_vivado > run_rf10.log 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
d=${1:?config folder, e.g. qat_w12_s6_seed0_rf10_vivado}
[ -d "$d" ] || { echo "no folder $d"; exit 1; }
if ! command -v vivado_hls > /dev/null; then
  source "${VIVADO_SETTINGS:-/opt/xilinx/Vivado/2020.1/settings64.sh}" > /dev/null 2>&1
fi
command -v vivado_hls > /dev/null || { echo "vivado_hls not on PATH (set VIVADO_SETTINGS to settings64.sh)"; exit 1; }
echo "$(date -u +%FT%TZ) $(hostname) $(vivado_hls -version 2>&1 | grep -m1 -i vivado)"
if [ -f "$d/synth_results.csv" ]; then echo "$(date -u +%FT%TZ) skip $d (done earlier)"; exit 0; fi
echo "$(date -u +%FT%TZ) start $d"
if ( cd "$d" && bash run_synth.sh > run_synth.out 2>&1 ); then
  echo "$(date -u +%FT%TZ) done $d"
else
  echo "$(date -u +%FT%TZ) FAILED $d (see $d/run_synth.out and $d/synth.log)"
fi
files=$(ls "$d"/synth_results.csv "$d"/synth.log "$d"/run_synth.out "$d"/hls_tool_version.txt 2>/dev/null)
tar czf "synth_results_${d}.tar.gz" $files
echo "$(date -u +%FT%TZ) finished; results: $HERE/synth_results_${d}.tar.gz"
