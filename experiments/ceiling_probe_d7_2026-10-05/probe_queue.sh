#!/usr/bin/env bash
# Created: 2026-10-06
# Last modified: 2026-10-06
#
# After C1 and D1 finish: S1 (2 x u160), D2 (u280 distilled from arm b), S2 (2 x u140), all d7
# 200M; then the queue's 9-9; then 9-S (2 x u280 at d9, 100M, 160 epochs).
# Memory rules (2026-10-06):
#   - a d7 200M job (and 9-S) starts only with >= 230 GB host memory available;
#   - a job of a new architecture must finish its first epoch, and its footprint is measured
#     (podman stats, written to footprints.tsv), before any other job starts alongside it;
#   - the queue driver is held (HOLD) while these jobs wait and load, and released for 9-9
#     after S2 has loaded; 9-S waits until 9-9 has finished its first epoch.
# Runs from ~/FNAL-AI4QC-GRU-probe-q (a copy of the code), so no file of a running job changes.
#
#     setsid nohup bash probe_queue.sh > probe_queue.out 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CD="$HOME/FNAL-AI4QC-GRU-probe-cd/experiments/ceiling_probe_d7_2026-10-05"
PB="$HOME/FNAL-AI4QC-GRU-probe-b/experiments/ceiling_probe_d7_2026-10-05"
Q="$HOME/FNAL-AI4QC-GRU-mwpm/experiments/mwpm_fixes_d9_2026-10"
NEED_GB=230
FOOT="$HERE/footprints.tsv"
log () { echo "[probe $(date -u +%FT%TZ)] $*" | tee -a "$HERE/probe.log"; echo "[mwpm-d9 $(date -u +%FT%TZ)] NOTE  $*" >> "$Q/driver.log"; }
avail () { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
swap_mb () { free -m | awk '/Swap/{print $3}'; }
loading () {  # a recently written run log with no epoch line yet = a job still loading
  local f
  for f in $(ls "$HERE"/*_2*/seed0/run.log "$CD"/*_2*/seed0/run.log "$PB"/*_2*/seed0/run.log "$Q"/9-*_2*/seed0/run.log 2>/dev/null); do
    [ -n "$(find "$f" -mmin -30)" ] && ! grep -q "\] epoch [0-9]*: val_loss" "$f" && return 0
  done
  return 1
}
hold () { echo "2026-10-06: $1" > "$Q/HOLD"; }
start () {  # start <arm>: wait for memory and no loading, start, wait for the first epoch, measure
  until [ "$(avail)" -ge "$NEED_GB" ] && ! loading; do sleep 120; done
  log "starting $1 (host available $(avail) GB >= $NEED_GB, swap $(swap_mb) MB)"
  (cd "$HERE" && setsid nohup bash run_arm.sh "$1" > "arm_$1.out" 2>&1 < /dev/null &)
  sleep 90
  local L m
  until L=$(ls "$HERE"/$1_*/seed0/run.log 2>/dev/null) && grep -qE "\] epoch 1: val_loss|Traceback|STOP" "$L"; do sleep 60; done
  m=$(podman stats --no-stream --format '{{.MemUsage}}' "probe_$1" 2>/dev/null | cut -d' ' -f1)
  [ -f "$FOOT" ] || echo -e "utc\tarm\tparams\tcontainer_mem_after_epoch_1\thost_available_gb\tswap_mb" > "$FOOT"
  echo -e "$(date -u +%FT%TZ)\t$1\t$(grep -oE 'params=[0-9,]+' "$L" | head -1)\t$m\t$(avail)\t$(swap_mb)" >> "$FOOT"
  log "$1 epoch 1 done: $(grep -E 'params=|teacher .*logits for all|\] epoch 1: val_loss|Traceback|STOP' "$L" | cut -c1-160 | tr '\n' ' '); footprint $m; host available $(avail) GB, swap $(swap_mb) MB"
}

hold "S1, D2, S2 (d7 200M) go before 9-9"
log "probe_queue: waiting for C1 and D1 to finish (HOLD keeps 9-9 back)"
until grep -q "arm c1 train-slice scored" "$CD/probe.log" && grep -q "arm d1 train-slice scored" "$CD/probe.log"; do sleep 120; done
log "C1 and D1 finished"
for arm in s1 d2 s2; do start "$arm"; done
rm -f "$Q/HOLD"
log "S1, D2, S2 running; HOLD removed so the queue can start 9-9 (9-S waits for 9-9's first epoch)"
until L=$(ls "$Q"/9-9_2*/seed0/run.log 2>/dev/null) && grep -q "\] epoch [0-9]*: val_loss" "$L"; do sleep 300; done
hold "9-S (d9 2 x u280) loading"
start 9s
rm -f "$Q/HOLD"
log "9-S running; HOLD removed"
