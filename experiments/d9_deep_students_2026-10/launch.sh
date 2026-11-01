#!/usr/bin/env bash
# Created: 2026-10-23
# Last modified: 2026-10-23
# Queue the three runs of prereg/d9_deep_students_2026-10.md, staggered so each start sees the
# previous one on the GPU.
#   setsid nohup bash launch.sh > runs/launch.out 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
STATUS="$HERE/status.log"
mkdir -p "$HERE/runs"
[ -f "$HERE/STOP" ] && { echo "STOP file present"; exit 1; }

RUNS=(kd9_10x70 kd9_8x70 hl9_10x70)
echo "$(date -u +%FT%TZ) launch: ${RUNS[*]} epochs $(cat "$HERE/epochs.txt" 2>/dev/null || echo 240)" >> "$STATUS"
for r in "${RUNS[@]}"; do
  setsid nohup bash "$HERE/run_student.sh" "$r" > "$HERE/runs/$r.out" 2>&1 < /dev/null &
  for _ in $(seq 60); do grep -q " $r started" "$STATUS" && break; sleep 5; done
  sleep 90
done
wait
echo "$(date -u +%FT%TZ) all runs finished" >> "$STATUS"
