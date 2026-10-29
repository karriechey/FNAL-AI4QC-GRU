#!/usr/bin/env bash
# Created: 2026-10-28
# Last modified: 2026-10-28
# The whole experiment as one queue (PLAN.md). Launch order: d5 LR scan, d5 L = 1-5, d7 scan,
# d7 L = 1-5, d9 scan, d9 L = 1-5, then second seeds at L_min - 1 and L_min for each d. Runs go
# up concurrently as resources allow; a launch waits only on what it needs (the L sweep on its
# d's LR rule, the seeds on its d's L_min). Safe to rerun after a crash or reboot: finished runs
# are skipped, unfinished ones resume from their last epoch snapshot.
#   setsid nohup bash queue.sh > queue/queue.out 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HOSTPY=${HOSTPY:-$HOME/tfenv/bin/python}
STATUS="$HERE/status.log"
SNAP_ROOT=${SNAP_ROOT:-/tmp/kchey_dvd}
mkdir -p "$HERE/queue" "$HERE/runs" "$SNAP_ROOT"
echo $$ > "$HERE/queue/queue.pid"
note () { echo "$(date -u +%FT%TZ) queue $*" >> "$STATUS"; }
dvd () { "$HOSTPY" "$HERE/dvd.py" "$@"; }
report () { flock "$HERE/.report.lock" "$HOSTPY" "$HERE/dvd.py" report > /dev/null 2>&1; }
finished () { dvd finished "$1"; }
running () { podman ps --format '{{.Names}}' | grep -qx "dvd_$1"; }

# /tmp is cleaned of files untouched for 10 days; the snapshots must outlive the queue
( while true; do find "$SNAP_ROOT" -type f -exec touch -a -m {} + 2>/dev/null; sleep 86400; done ) &
KEEPALIVE=$!
trap 'kill $KEEPALIVE 2>/dev/null' EXIT

declare -A PIDS
start () {   # launch one run in the background, return once it has started or ended
  local r=$1
  [ -f "$HERE/STOP" ] && { note "STOP file present, $r not launched"; return 1; }
  if finished "$r"; then note "$r already finished, skipped"; return 0; fi
  if running "$r"; then note "$r already running"; return 0; fi
  echo "$r" >> "$HERE/queue/planned.txt"
  # one automatic retry (resume) when a run fails
  nohup bash -c "bash '$HERE/run_dvd.sh' '$r' || { sleep 600; bash '$HERE/run_dvd.sh' '$r'; }" \
    >> "$HERE/queue/$r.out" 2>&1 < /dev/null &
  PIDS[$r]=$!
  local t0=$(date +%s)
  until grep -qE " $r (started|not started|FAILED)" <(tail -n 200 "$STATUS" 2>/dev/null) \
        || ! kill -0 ${PIDS[$r]} 2>/dev/null; do sleep 30; done
  sleep 90                 # the GPU allocation shows up before the next resource check
}

wait_runs () {   # until every named run is finished (DONE / KILLED / DIVERGED) or its launcher has exited
  local r all
  while true; do
    all=1
    for r in "$@"; do
      finished "$r" && continue
      [ -n "${PIDS[$r]:-}" ] && kill -0 ${PIDS[$r]} 2>/dev/null && { all=0; continue; }
      running "$r" && { all=0; continue; }
      # not finished and nothing driving it: launched by an earlier queue that died, or failed twice
      [ -z "${PIDS[$r]:-}" ] && { start "$r"; all=0; continue; }
    done
    [ $all = 1 ] && return
    sleep 300
  done
}

lr_gate () {   # LR rule for d, with lrscan_d7's conditional 5e-4 runs; sets LR (empty when none)
  local d=$1 need
  wait_runs $(dvd scan-names "$d")
  dvd lr-rule "$d" > /dev/null
  need=$("$HOSTPY" -c "import json,sys; print(' '.join(json.load(open(sys.argv[1])).get('need', [])))" "$HERE/lr/d$d.json")
  if [ -n "$need" ]; then
    note "d$d: no grid LR stable at L = 1 and 5, running $need"
    for r in $need; do start "$r"; done
    wait_runs $need
    dvd lr-rule "$d" > /dev/null
  fi
  LR=$("$HOSTPY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('lr') or '')" "$HERE/lr/d$d.json")
}

note "start (pid $$)"
{ cat "$HERE/queue/planned.txt" 2>/dev/null
  for d in 5 7 9; do dvd scan-names "$d" | tr ' ' '\n'; for L in 5 4 3 2 1; do dvd sweep-name "$d" "$L"; done; done
} | awk 'NF && !seen[$0]++' > "$HERE/queue/planned.tmp" && mv "$HERE/queue/planned.tmp" "$HERE/queue/planned.txt"
report
declare -A SWEEP_OK
for d in 5 7 9; do
  for r in $(dvd scan-names "$d"); do start "$r" || exit 1; done
  lr_gate "$d"
  if [ -z "$LR" ]; then
    note "d$d: no stable LR, L sweep and seeds for d$d not run ($(grep -o '"reason": "[^"]*"' "$HERE/lr/d$d.json"))"
    SWEEP_OK[$d]=0
    report; continue
  fi
  note "d$d peak LR $LR"
  SWEEP_OK[$d]=1
  for L in 5 4 3 2 1; do start "$(dvd sweep-name "$d" "$L")" || exit 1; done
  report
done

for d in 5 7 9; do
  [ "${SWEEP_OK[$d]}" = 1 ] || continue
  wait_runs $(for L in 1 2 3 4 5; do dvd sweep-name "$d" "$L"; done)
  dvd lmin "$d" > /dev/null
  note "d$d L_min: $(cat "$HERE/lmin/d$d.json" | tr -d '\n ' | cut -c1-200)"
  report
  for r in $("$HOSTPY" -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['seed_runs']))" "$HERE/lmin/d$d.json"); do
    start "$r" || exit 1
  done
done

wait_runs $(cat "$HERE/queue/planned.txt" 2>/dev/null | sort -u)
for d in 5 7 9; do [ "${SWEEP_OK[$d]}" = 1 ] && dvd lmin "$d" > /dev/null; done
report
"$HOSTPY" "$HERE/../../figures/fig_depth_vs_distance.py" --root "$HERE" > "$HERE/queue/figure.out" 2>&1
note "figure exit $?"
note "finished"
