#!/usr/bin/env bash
# Created: 2026-10-18
# Last modified: 2026-10-18
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/runs"
for r in kd5_u140 hl5_u140 kd5_u70 hl5_u70; do
  setsid nohup bash "$HERE/run_kd.sh" "$r" > "$HERE/runs/$r.out" 2>&1 < /dev/null &
  sleep 20                    # let the launcher's free-GPU check see the previous job
done
