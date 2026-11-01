#!/usr/bin/env bash
# Created: 2026-10-21
# Last modified: 2026-10-21
# Part B: waits for the sweep LR (lr_scan/sweep_lr.json), then for depth_sweep/GO (written once
# the prereg amendment is committed; proceeds after GO_WAIT_H hours without it), then queues
# the depth sweep and the parameter-matched single layers, deepest first. report.py at the end.
#   setsid nohup bash launch_depth_sweep.sh > depth_sweep/launch.out 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HOSTPY=${HOSTPY:-$HOME/tfenv/bin/python}
STATUS="$HERE/status.log"
GO_WAIT_H=${GO_WAIT_H:-3}
mkdir -p "$HERE/depth_sweep"
note () { echo "$(date -u +%FT%TZ) depth_sweep $*" >> "$STATUS"; }
field () { "$HOSTPY" -c "import json,sys; v=json.load(open(sys.argv[1])).get(sys.argv[2]); print('' if v is None else v)" "$HERE/lr_scan/sweep_lr.json" "$1" 2>/dev/null; }

until grep -q "lr_scan Part A finished" "$STATUS" 2>/dev/null; do sleep 300; done
[ -f "$HERE/STOP" ] && { note "STOP file present, depth sweep not launched"; exit 1; }
LR=$(field sweep_lr)
[ -n "$LR" ] || { note "no sweep LR in lr_scan/sweep_lr.json, depth sweep not launched"; exit 1; }
note "sweep LR $LR; waiting up to ${GO_WAIT_H} h for depth_sweep/GO"
t0=$(date +%s)
until [ -f "$HERE/depth_sweep/GO" ] || [ $(( $(date +%s) - t0 )) -ge $((GO_WAIT_H * 3600)) ]; do sleep 60; done
[ -f "$HERE/depth_sweep/GO" ] && note "GO present" || note "no GO after ${GO_WAIT_H} h; launching (amendment committed later)"
echo "$LR" > "$HERE/depth_sweep/sweep_lr.txt"

RUNS=(depth_d7_L10_u70 depth_d7_L09_u70 depth_d7_L08_u70 depth_d7_L07_u70 depth_d7_L06_u70 depth_d7_L05_u70
      depth_d7_L04_u70 depth_d7_L03_u70)
[ "$LR" = 1e-3 ] || RUNS+=(depth_d7_L02_u70)
RUNS+=(depth_d7_L01_u70 match_d7_L08_u255 match_d7_L04_u172)
note "launch: ${RUNS[*]}"
for r in "${RUNS[@]}"; do
  [ -f "$HERE/STOP" ] && { note "STOP file present, no further launches"; break; }
  nohup bash "$HERE/run_depth.sh" "$r" > "$HERE/depth_sweep/$r.out" 2>&1 < /dev/null &
  pid=$!
  until grep -q " $r \(started\|not started\)" "$STATUS" 2>/dev/null || ! kill -0 $pid 2>/dev/null; do sleep 30; done
  sleep 90
done
wait
"$HOSTPY" "$HERE/report.py" > "$HERE/depth_sweep/report.out" 2>&1
note "Part B finished; report.py exit $?"
