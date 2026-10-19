#!/usr/bin/env bash
# Created: 2026-10-19
# Last modified: 2026-10-19
# Waits for the d5 e160 pair, writes the d5 report, then starts d9 (hl9_3x70 behind its gate).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HOSTPY=${HOSTPY:-$HOME/tfenv/bin/python}
log () { echo "[kd $(date -u +%FT%TZ)] $*" | tee -a "$HERE/runs/pilot.log"; }
while pgrep -f 'run_kd.sh [kh][dl]5_u70_e160' > /dev/null; do sleep 300; done
for r in kd5_u70_e160 hl5_u70_e160; do
  ls "$HERE"/runs/${r}_*/DONE > /dev/null 2>&1 || { log "after_e160: $r has no DONE, d9 not started. STOP."; exit 1; }
done
"$HOSTPY" "$HERE/report.py" --d 5 > "$HERE/runs/report_d5.log" 2>&1
log "after_e160: d5 report exit $?; starting d9"
bash "$HERE/launch.sh" kd9_3x140 kd9_1x140
setsid nohup bash -c "bash '$HERE/run_kd.sh' kd9_3x70 > '$HERE/runs/kd9_3x70.out' 2>&1; bash '$HERE/run_kd.sh' hl9_3x70 > '$HERE/runs/hl9_3x70.out' 2>&1" < /dev/null > /dev/null 2>&1 &
