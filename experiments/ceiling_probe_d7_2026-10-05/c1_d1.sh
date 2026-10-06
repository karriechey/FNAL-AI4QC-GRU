#!/usr/bin/env bash
# Created: 2026-10-06
# Last modified: 2026-10-06
#
# Start C1 (u280 control) and D1 (u280 distilled from arm a) as soon as host memory allows,
# ahead of the queue's 9-7 and 9-9 (held by HOLD in the queue folder). C1 first; D1 once C1
# has loaded and memory still allows (together if both fit, never into swap). When both have
# loaded, HOLD is removed so the queue continues. Runs from ~/FNAL-AI4QC-GRU-probe-cd, a copy
# of the code with --teacher-model, so no file of a running job is changed.
#
#     setsid nohup bash c1_d1.sh > c1_d1.out 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
Q="$HOME/FNAL-AI4QC-GRU-mwpm/experiments/mwpm_fixes_d9_2026-10"
NEED_GB=210            # arm a peaked at 167 GB on the same data; margin for the load
log () { echo "[probe $(date -u +%FT%TZ)] $*" | tee -a "$HERE/probe.log"; echo "[mwpm-d9 $(date -u +%FT%TZ)] NOTE  $*" >> "$Q/driver.log"; }
avail () { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
swap_mb () { free -m | awk '/Swap/{print $3}'; }
loading () {  # any container of ours with no epoch line in its run log yet
  local f
  for f in $(ls "$HERE"/*_2*/seed0/run.log "$Q"/9-*_2*/seed0/run.log $HOME/FNAL-AI4QC-GRU-probe-b/experiments/ceiling_probe_d7_2026-10-05/b_*_2*/seed0/run.log 2>/dev/null); do
    [ -n "$(find "$f" -mmin -30)" ] && ! grep -q "\] epoch [0-9]*: val_loss" "$f" && return 0
  done
  return 1
}
start () {  # start <arm>
  until [ "$(avail)" -ge "$NEED_GB" ] && ! loading; do sleep 120; done
  log "starting $1 (host available $(avail) GB, swap $(swap_mb) MB)"
  (cd "$HERE" && setsid nohup bash run_arm.sh "$1" > "arm_$1.out" 2>&1 < /dev/null &)
  sleep 90
  local L
  until L=$(ls "$HERE"/$1_*/seed0/run.log 2>/dev/null) && grep -qE "\] epoch 1: val_loss|Traceback|STOP" "$L"; do sleep 60; done
  log "$1 loaded: $(grep -E 'params=|teacher .*logits for all|\] epoch 1: val_loss|Traceback|STOP' "$L" | cut -c1-200 | tr '\n' ' '); host available $(avail) GB, swap $(swap_mb) MB"
}
log "c1_d1: waiting for host memory >= $NEED_GB GB (HOLD keeps 9-7 and 9-9 back)"
start c1
start d1
rm -f "$Q/HOLD"
log "C1 and D1 both running; HOLD removed, the queue continues with 9-7 and 9-9"
