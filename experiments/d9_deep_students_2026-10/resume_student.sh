#!/usr/bin/env bash
# Created: 2026-10-24
# Last modified: 2026-10-24
# Continue a run that died mid-training (grace1 rebooted on 2026-10-22 and killed six runs of
# the d=7 depth sweep). Resumes in the run's own folder from its last complete epoch snapshot,
# which carries the Adam state, so the shot stream and the schedule continue unchanged.
#   bash resume_student.sh kd9_10x70
# A run that has not reached SAVE_EPOCHS_FROM (epoch 40) has no snapshot and must be restarted.
set -uo pipefail
RUN=$1
HERE="$(cd "$(dirname "$0")" && pwd)"
DEPTH="$HERE/../depth_scaling_d7_2026-10"
HOSTPY=${HOSTPY:-$HOME/tfenv/bin/python}

OUT=$(ls -d "$HERE"/runs/${RUN}_2026* 2>/dev/null | tail -1)
[ -n "$OUT" ] || { echo "no run folder for $RUN"; exit 1; }
[ -f "$OUT/seed0/COMPLETE" ] && { echo "$OUT already COMPLETE"; exit 1; }
podman ps --format '{{.Names}}' | grep -qx "d9_$RUN" && { echo "d9_$RUN is still running"; exit 1; }

DONE_EP=$(grep -cE '^[0-9]+/[0-9]+ - [0-9]+s - loss' "$OUT/seed0/run.log")
W=$(printf '%s/seed0/epochs/epoch%03d.weights.h5' "$OUT" "$DONE_EP")
[ -s "$W" ] || { echo "no epoch $DONE_EP snapshot at $W (epochs are saved from 40); restart instead"; exit 1; }
CUDA_VISIBLE_DEVICES= "$HOSTPY" -c "
import h5py, sys
with h5py.File(sys.argv[1]) as h:
    assert 'optimizer' in h, 'no optimizer state'
" "$W" || { echo "$W unreadable or without Adam state"; exit 1; }

[ -f "$OUT/seed0/run_part1.log" ] || cp "$OUT/seed0/run.log" "$OUT/seed0/run_part1.log"
"$HOSTPY" "$DEPTH/history_from_log.py" --log "$OUT/seed0/run_part1.log" --epochs "$DONE_EP" \
  --out "$OUT/seed0/resume_history.json" || exit 1

export RESUME_WEIGHTS="$W" INITIAL_EPOCH="$DONE_EP" RESUME_OUT="$OUT"
export RESUME_BEST_JSON=$(ls "$OUT"/seed0/ckpt/*.best.json 2>/dev/null | head -1)
export RESUME_HISTORY="$OUT/seed0/resume_history.json"
echo "$(date -u +%FT%TZ) $RUN resuming from epoch $DONE_EP in $(basename "$OUT")" >> "$HERE/status.log"
exec bash "$HERE/run_student.sh" "$RUN"
