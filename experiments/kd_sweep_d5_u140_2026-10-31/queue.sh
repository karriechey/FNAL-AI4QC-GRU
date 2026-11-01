#!/usr/bin/env bash
# Created: 2026-10-31 | Last updated: 2026-10-31
# Starts the cells of configs.tsv marked "new" while the GPU has room: a cell starts when at least
# NEED_MIB is free on the card, then the queue waits GAP_S so the new process's memory shows up before the
# next check. Resume-safe: cells with keep/<run>_*/DONE or a live run_one.sh are skipped; a cell is tried at
# most 3 times. Stop with: touch STOP
#   setsid nohup bash queue.sh > queue.out 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NEED_MIB=${NEED_MIB:-12000}; GAP_S=${GAP_S:-300}
LOG=/tmp/kchey_kdsweep/runs/sweep.log
mkdir -p /tmp/kchey_kdsweep/runs "$HERE/logs"
say () { echo "[queue $(date -u +%FT%TZ)] $*"; }

running () { pgrep -f "run_one.sh $1 $2\$" > /dev/null; }
attempts () { cat "$LOG" 2>/dev/null | grep -c "START $1 "; }

while [ ! -f "$HERE/STOP" ]; do
  pending=0
  while read -r name alpha temp source; do
    [ "$source" = new ] || continue
    ls "$HERE"/keep/"${name}"_*/DONE > /dev/null 2>&1 && continue
    running "$alpha" "$temp" && { pending=1; continue; }
    [ "$(attempts "$name")" -ge 3 ] && { say "$name tried 3 times, skipped"; continue; }
    pending=1
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "${free:-0}" -ge "$NEED_MIB" ] || continue
    say "start $name (alpha $alpha, T $temp), ${free} MiB free"
    setsid nohup bash "$HERE/run_one.sh" "$alpha" "$temp" > "$HERE/logs/$name.out" 2>&1 < /dev/null &
    sleep "$GAP_S"
  done < <(grep -v '^#' "$HERE/configs.tsv")
  [ "$pending" = 0 ] && { say "all cells done or given up"; break; }
  sleep 120
done
