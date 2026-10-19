#!/usr/bin/env bash
# Created: 2026-10-19
# Last modified: 2026-10-19
#   bash launch.sh kd7_2x140 hl7_2x140 ...
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/runs"
for r in "$@"; do
  setsid nohup bash "$HERE/run_kd.sh" "$r" > "$HERE/runs/$r.out" 2>&1 < /dev/null &
  sleep 20                    # let the launcher's free-GPU check see the previous job
done
