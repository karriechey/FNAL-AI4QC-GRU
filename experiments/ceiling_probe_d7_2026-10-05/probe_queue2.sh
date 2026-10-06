#!/usr/bin/env bash
# Created: 2026-10-06
# Last modified: 2026-10-06
#
# Takes over from probe_queue.sh (stopped by this script) with the memory rule chosen on
# 2026-10-06: a job starts when host memory available >= the measured footprint of the same
# kind of job + 40 GB (footprints.tsv: container memory after epoch 1), instead of a flat
# 230 GB. A job with no measured reference still needs 230 GB. A new architecture must still
# finish its first epoch (and be measured) before another job starts alongside it.
#   D2 (already started): wait for its first epoch, measure.
#   S2: reference S1 (same kind: stacked GRU, d7 200M).
#   then HOLD removed so the queue starts 9-9 (its threshold is in jobs.tsv: 9-7's 130 GB + 40).
#   9-S: reference 9-7 (d9 100M, 130 GB measured) + 40 GB; starts after 9-9's first epoch.
#
#     setsid nohup bash probe_queue2.sh > probe_queue2.out 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CD="$HOME/FNAL-AI4QC-GRU-probe-cd/experiments/ceiling_probe_d7_2026-10-05"
PB="$HOME/FNAL-AI4QC-GRU-probe-b/experiments/ceiling_probe_d7_2026-10-05"
Q="$HOME/FNAL-AI4QC-GRU-mwpm/experiments/mwpm_fixes_d9_2026-10"
FOOT="$HERE/footprints.tsv"
MARGIN=40
log () { echo "[probe $(date -u +%FT%TZ)] $*" | tee -a "$HERE/probe.log"; echo "[mwpm-d9 $(date -u +%FT%TZ)] NOTE  $*" >> "$Q/driver.log"; }
avail () { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
swap_mb () { free -m | awk '/Swap/{print $3}'; }
loading () {
  local f
  for f in $(ls "$HERE"/*_2*/seed0/run.log "$CD"/*_2*/seed0/run.log "$PB"/*_2*/seed0/run.log "$Q"/9-*_2*/seed0/run.log 2>/dev/null); do
    [ -n "$(find "$f" -mmin -30)" ] && ! grep -q "\] epoch [0-9]*: val_loss" "$f" && return 0
  done
  return 1
}
foot_gb () { awk -F'\t' -v a="$1" '$2 == a {v = $4} END {sub(/GB/, "", v); printf "%d", v + 0.5}' "$FOOT"; }
hold () { echo "2026-10-06: $1" > "$Q/HOLD"; }
measure () {  # measure <arm>: wait for epoch 1, record the container footprint
  local L m
  until L=$(ls "$HERE"/$1_*/seed0/run.log 2>/dev/null) && grep -qE "\] epoch 1: val_loss|Traceback|STOP" "$L"; do sleep 60; done
  m=$(podman stats --no-stream --format '{{.MemUsage}}' "probe_$1" 2>/dev/null | cut -d' ' -f1)
  echo -e "$(date -u +%FT%TZ)\t$1\t$(grep -oE 'params=[0-9,]+' "$L" | head -1)\t$m\t$(avail)\t$(swap_mb)" >> "$FOOT"
  log "$1 epoch 1 done: $(grep -E 'params=|\] epoch 1: val_loss|Traceback|STOP' "$L" | cut -c1-140 | tr '\n' ' '); footprint $m; host available $(avail) GB, swap $(swap_mb) MB"
}
start () {  # start <arm> <need_gb> <why>
  until [ "$(avail)" -ge "$2" ] && ! loading; do sleep 120; done
  log "starting $1 (host available $(avail) GB >= $2 = $3; swap $(swap_mb) MB)"
  (cd "$HERE" && setsid nohup bash run_arm.sh "$1" > "arm_$1.out" 2>&1 < /dev/null &)
  sleep 90
  measure "$1"
}

for p in $(ps -eo pid,args | awk '$2 == "bash" && $3 == "probe_queue.sh" {print $1}'); do kill "$p"; done
log "probe_queue2: probe_queue.sh stopped; memory rule now = measured footprint of the same kind of job + ${MARGIN} GB (230 GB without a reference)"
grep -qP '\td2\t' "$FOOT" || measure d2
S1=$(foot_gb s1)
start s2 $((S1 + MARGIN)) "S1 footprint ${S1} GB + ${MARGIN}"
rm -f "$Q/HOLD"
log "S1, D2, S2 running; HOLD removed so the queue can start 9-9"
until L=$(ls "$Q"/9-9_2*/seed0/run.log 2>/dev/null) && grep -q "\] epoch [0-9]*: val_loss" "$L"; do sleep 300; done
hold "9-S (d9 2 x u280) loading"
start 9s $((130 + MARGIN)) "9-7 footprint 130 GB + ${MARGIN}"
rm -f "$Q/HOLD"
log "9-S running; HOLD removed"
