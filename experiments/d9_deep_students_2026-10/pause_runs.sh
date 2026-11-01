#!/usr/bin/env bash
# Created: 2026-10-24
# Last modified: 2026-10-24
# Pause runs at the next epoch boundary, as experiments/ceiling_probe_d7_2026-10-05/
# pause_probe.sh does. Waits for the next "epoch N: global grad norm" line, which
# train_student.py prints after that epoch's checkpoints are written, so the snapshot and
# .lastepoch.weights.h5 hold the model and the Adam state including the update count. Stops the
# run_student.sh wrapper first, so it cannot go on to score a half-trained checkpoint, then the
# container, then writes PAUSED. Continue with resume_student.sh.
#   bash pause_runs.sh kd9_10x70 hl9_10x70 kd9_8x70
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
STATUS="$HERE/status.log"
HOSTPY=${HOSTPY:-$HOME/tfenv/bin/python}
DEPTH="$HERE/../depth_scaling_d7_2026-10"

for RUN in "$@"; do
  OUT=$(ls -d "$HERE"/runs/${RUN}_2026* 2>/dev/null | tail -1)
  [ -n "$OUT" ] || { echo "no run folder for $RUN"; continue; }
  L="$OUT/seed0/run.log"
  n0=$(grep -c "global grad norm" "$L")
  echo "$(date -u +%FT%TZ) $RUN pause requested; waiting for the end of epoch $((n0 + 1))" >> "$STATUS"
  until [ "$(grep -c "global grad norm" "$L")" -gt "$n0" ]; do sleep 10; done
  for p in $(pgrep -f "[r]un_student.sh $RUN\$"); do kill "$p" 2>/dev/null; done
  podman stop -t 60 "d9_$RUN" > /dev/null 2>&1
  N=$(grep -c "global grad norm" "$L")
  W="$OUT/seed0/epochs/epoch$(printf '%03d' "$N").weights.h5"
  # the history of the finished epochs, for --resume-history
  cp "$L" "$OUT/seed0/run_part1.log"
  "$HOSTPY" "$DEPTH/history_from_log.py" --log "$OUT/seed0/run_part1.log" --epochs "$N" \
    --out "$OUT/seed0/resume_history.json" >> "$STATUS" 2>&1
  { echo "paused after epoch $N at $(date -u +%FT%TZ)"
    echo "snapshot $W $( [ -s "$W" ] && echo present || echo MISSING)"
    echo "lastepoch $(ls "$OUT"/seed0/ckpt/*.lastepoch.weights.h5 2>/dev/null | head -1)"; } > "$OUT/PAUSED"
  echo "$(date -u +%FT%TZ) $RUN paused after epoch $N" >> "$STATUS"
done
