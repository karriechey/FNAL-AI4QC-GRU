#!/usr/bin/env bash
# Created: 2026-10-29
# Last modified: 2026-10-29
# X-basis teachers end to end (PLAN.md): pools d5, d7, d9 in turn, each teacher launched as soon
# as its pool exists, report when all three are finished, then the paused depth_vs_distance
# queue is resumed. Safe to rerun: existing pools and finished runs are skipped.
#   setsid nohup bash queue_x.sh > queue_x.out 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HOSTPY=${HOSTPY:-$HOME/tfenv/bin/python}
DVD="$(cd "$HERE/../depth_vs_distance" && pwd)"
STATUS="$HERE/status.log"
note () { echo "$(date -u +%FT%TZ) queue $*" >> "$STATUS"; }
finished () { local o; o=$(ls -d "$HERE"/runs/$1_2* 2>/dev/null | tail -1); [ -n "$o" ] && ls "$o"/DONE "$o"/KILLED "$o"/DIVERGED > /dev/null 2>&1; }
note "start (pid $$)"

( while true; do find /tmp/kchey_xbasis -type f -exec touch -a -m {} + 2>/dev/null; sleep 86400; done ) &
KEEPALIVE=$!
trap 'kill $KEEPALIVE 2>/dev/null' EXIT

declare -A PIDS
RUNS=(x5_u280 x7_2xu280 x9_3xu280)
for i in 0 1 2; do
  d=$((5 + 2 * i)); r=${RUNS[$i]}
  bash "$HERE/gen_pools.sh" $d
  ls /tmp/kchey_xbasis/pools/data_d${d}_p0.004_r${d}_X_FORMAL_seed*.fingerprint.json > /dev/null 2>&1 \
    || { note "d$d pool missing after generation (gen_pools.log); $r not run"; continue; }
  finished "$r" && { note "$r already finished"; continue; }
  nohup bash -c "bash '$HERE/run_x.sh' $r || { sleep 600; bash '$HERE/run_x.sh' $r; }" >> "$HERE/runs_$r.out" 2>&1 < /dev/null &
  PIDS[$r]=$!
  note "$r launched"
done
for r in "${!PIDS[@]}"; do wait ${PIDS[$r]}; done
flock "$HERE/.report.lock" "$HOSTPY" "$HERE/report_x.py" > "$HERE/report_x.out" 2>&1
note "report exit $?"

rm -f "$DVD/STOP"
echo "$(date -u +%FT%TZ) queue resumed after the X-basis teachers" >> "$DVD/status.log"
( cd "$DVD" && setsid nohup bash queue.sh >> queue/queue.out 2>&1 < /dev/null & )
note "depth_vs_distance queue resumed; finished"
