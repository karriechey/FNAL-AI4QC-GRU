#!/usr/bin/env bash
# Created: 2026-10-21
# Last modified: 2026-10-21
# Part A: the eight LR-scan runs, then the sweep LR rule (classify_lr_scan.py); adds the
# conditional L=10 5e-4 run when L=10 at 1e-3 is not stable.
#   setsid nohup bash launch_lr_scan.sh > lr_scan/launch.out 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HOSTPY=${HOSTPY:-$HOME/tfenv/bin/python}
STATUS="$HERE/status.log"
mkdir -p "$HERE/lr_scan"
note () { echo "$(date -u +%FT%TZ) lr_scan $*" >> "$STATUS"; }
stop_top () {
  { echo "STOP $(date -u +%FT%TZ): $*"; cat "$STATUS"; } > "$STATUS.tmp" && mv "$STATUS.tmp" "$STATUS"
  touch "$HERE/STOP"; note "Part A finished (stopped)"; exit 1
}

start () {   # one run, detached from this shell's fate; returns once it has started or ended
  local r=$1
  nohup bash "$HERE/run_depth.sh" "$r" > "$HERE/lr_scan/$r.out" 2>&1 < /dev/null &
  local pid=$!
  until grep -q " $r \(started\|not started\)" "$STATUS" 2>/dev/null || ! kill -0 $pid 2>/dev/null; do sleep 30; done
  sleep 90                 # the GPU allocation shows up before the next resource check
}

note "launch: 8 runs, L=1 and 10 x LR 1e-3, 2e-3, 3e-3, 5e-3"
for L in 10 01; do
  for lr in 1e-3 2e-3 3e-3 5e-3; do
    [ -f "$HERE/STOP" ] && { note "STOP file present, no further launches"; exit 1; }
    start "lrscan_d7_L${L}_lr$lr"
  done
done
wait
"$HOSTPY" "$HERE/classify_lr_scan.py" > "$HERE/lr_scan/classify.out" 2>&1 || stop_top "LR-scan classification failed, see lr_scan/classify.out"
if "$HOSTPY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('need') else 1)" "$HERE/lr_scan/sweep_lr.json"; then
  note "L=10 at 1e-3 not stable: conditional run lrscan_d7_L10_lr5e-4"
  bash "$HERE/run_depth.sh" lrscan_d7_L10_lr5e-4 > "$HERE/lr_scan/lrscan_d7_L10_lr5e-4.out" 2>&1 < /dev/null
  "$HOSTPY" "$HERE/classify_lr_scan.py" > "$HERE/lr_scan/classify.out" 2>&1 || stop_top "LR-scan classification failed, see lr_scan/classify.out"
fi
note "rule: $(cat "$HERE/lr_scan/sweep_lr.json" | tr -d '\n ')"
if "$HOSTPY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('stop') else 1)" "$HERE/lr_scan/sweep_lr.json"; then
  stop_top "neither 1e-3 nor 5e-4 is stable at L=10; depth sweep not run"
fi
note "Part A finished"
