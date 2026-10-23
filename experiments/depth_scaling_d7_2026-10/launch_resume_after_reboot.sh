#!/usr/bin/env bash
# Created: 2026-10-23
# Last modified: 2026-10-23
# Continue the six depth-sweep runs the 2026-10-22 21:19Z reboot killed, each in its own
# folder from its .lastepoch checkpoint, then report.
#   setsid nohup bash launch_resume_after_reboot.sh > depth_sweep/resume.out 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
STATUS="$HERE/status.log"
HOSTPY=${HOSTPY:-$HOME/tfenv/bin/python}
# another user's llama-server parks 57 GB on the card at 0% utilization (README)
export FOREIGN_MAX_MIB=${FOREIGN_MAX_MIB:-70000}

top () { echo "$(date -u +%FT%TZ) resume $*" >> "$STATUS"; }
[ -f "$HERE/STOP" ] && { top "not started: STOP file present"; exit 1; }

RUNS=(depth_d7_L10_u70 depth_d7_L09_u70 depth_d7_L08_u70 depth_d7_L07_u70
      depth_d7_L06_u70 depth_d7_L05_u70)
top "launch: ${RUNS[*]}"
for r in "${RUNS[@]}"; do
  out=$(ls -d "$HERE"/depth_sweep/${r}_2026* 2>/dev/null | tail -1)
  [ -n "$out" ] || { top "$r: no run folder, skipped"; continue; }
  [ -f "$out/seed0/COMPLETE" ] && { top "$r: already COMPLETE, skipped"; continue; }
  RESUME_OUT="$out" setsid nohup bash "$HERE/run_depth.sh" "$r" \
    > "$HERE/depth_sweep/${r}_resume.out" 2>&1 < /dev/null &
  for _ in $(seq 60); do grep -q "^.* $r started" "$STATUS" && break; sleep 5; done
  sleep 90
done
wait
top "all resumed runs finished; running report.py"
"$HOSTPY" "$HERE/report.py" >> "$HERE/depth_sweep/report.out" 2>&1
top "report.py exit $?"
