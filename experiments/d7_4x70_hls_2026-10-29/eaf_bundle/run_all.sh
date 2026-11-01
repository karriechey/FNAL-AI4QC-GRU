#!/usr/bin/env bash
# Created: 2026-10-29 | Last updated: 2026-10-29
# d7 4 x GRU 70 on EAF, Vivado HLS 2020.1 only (no pip, no network):
#   1 check that the part exists in this Vivado HLS install, stop if not
#   2 per project: C-sim on the 100-shot testbench, compare with the expected output (grace1 hls4ml C-sim),
#     then C synthesis (skipped for a project whose C-sim does not match)
#   3 collect reports and logs into results_d7_4x70_<timestamp>.tar.gz
#   bash run_all.sh                       all four projects, one after another
#   bash run_all.sh P16_t4096 P8_t4096    a subset (one server per subset)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PART="xcvu13p-flga2577-2-e"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p logs
LOG="logs/run_all_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1
say() { echo "$(date -u +%FT%TZ) $*"; }

if ! command -v vivado_hls > /dev/null; then
  set +u; source "${VIVADO_SETTINGS:-/opt/xilinx/Vivado/2020.1/settings64.sh}" > /dev/null 2>&1; set -u
fi
command -v vivado_hls > /dev/null || { say "STOP: vivado_hls not on PATH (set VIVADO_SETTINGS to settings64.sh)"; exit 1; }
say "host $(hostname)  $(vivado_hls -version 2>&1 | grep -m1 -i vivado)"

# 1 part check
rm -rf .partcheck && mkdir .partcheck && ( cd .partcheck && printf '%s\n' \
  'open_project -reset partcheck' 'open_solution -reset s' "set_part {$PART}" 'exit' > p.tcl && \
  vivado_hls -f p.tcl > ../logs/part_check.log 2>&1 )
if grep -q "ERROR" logs/part_check.log || ! grep -qi "$PART" logs/part_check.log; then
  say "STOP: part $PART is not available in this Vivado HLS install. See logs/part_check.log:"
  grep -i "error\|part" logs/part_check.log | head -5
  exit 1
fi
say "part $PART found"

# peak memory of the Vivado HLS processes, sampled every 15 s
peak_rss_mb() { local pid=$1 peak=0 cur; while kill -0 "$pid" 2>/dev/null; do
  cur=$(ps -C vivado_hls -o rss= 2>/dev/null | awk '{s+=$1} END {print int(s/1024)}'); [ "${cur:-0}" -gt "$peak" ] && peak=$cur
  sleep 15; done; echo "$peak"; }

run_vivado() {   # project, step name, csim, synth
  local p=$1 step=$2 csim=$3 synth=$4 reset=$5
  printf 'array set opt {\n    reset %s\n    csim %s\n    synth %s\n    cosim 0\n    validation 0\n    export 0\n    vsynth 0\n    fifo_opt 0\n    pnr 0\n}\n' \
    "$reset" "$csim" "$synth" > "projects/$p/build_opt.tcl"
  local t0=$(date +%s)
  ( cd "projects/$p" && vivado_hls -f build_prj.tcl > "$HERE/logs/${p}_${step}.log" 2>&1 ) &
  local pid=$!
  local peak; peak=$(peak_rss_mb $pid)
  wait $pid; local rc=$?
  echo "{\"project\": \"$p\", \"step\": \"$step\", \"rc\": $rc, \"wall_s\": $(( $(date +%s) - t0 )), \"peak_rss_mb_sampled\": $peak}" \
    > "logs/${p}_${step}_cost.json"
  say "$p $step rc=$rc wall $(( $(date +%s) - t0 )) s peak ~${peak} MB"
  return $rc
}

PROJECTS=("$@")
[ ${#PROJECTS[@]} -gt 0 ] || PROJECTS=(P16_t4096 P8_t4096 P16_tdefault P8_tdefault)
for p in "${PROJECTS[@]}"; do
  [ -d "projects/$p" ] || { say "no project $p, skipped"; continue; }
  say "start $p"
  run_vivado "$p" csim 1 0 1
  if python3 compare_csim.py "projects/$p" > "logs/${p}_csim_compare.json"; then
    say "$p C-sim matches the expected 100 outputs: $(cat logs/${p}_csim_compare.json | tr -d '\n' | cut -c1-200)"
    run_vivado "$p" csynth 0 1 0 || say "$p csynth FAILED (see logs/${p}_csynth.log)"
  else
    say "$p C-sim does NOT match the expected outputs, csynth skipped: $(cat logs/${p}_csim_compare.json | tr -d '\n' | cut -c1-300)"
  fi
  say "done $p"
done

# 3 collect
python3 collect.py > "logs/results_table_${STAMP}.txt"
files=$(ls logs/* results_table.csv projects/*/myproject_prj/solution1/syn/report/myproject_csynth.* \
        projects/*/tb_data/csim_results.log projects/*/firmware/parameters.h projects/*/firmware/defines.h \
        projects/*/hls4ml_config_used.json 2>/dev/null)
tar czf "results_d7_4x70_${STAMP}.tar.gz" $files
say "finished; results: $HERE/results_d7_4x70_${STAMP}.tar.gz"
cat "logs/results_table_${STAMP}.txt"
