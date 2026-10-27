#!/usr/bin/env bash
# Created: 2026-10-24
# Last modified: 2026-10-27 (epoch from the printed epoch number; rotate run.log into a new part)
# Continue a run that was paused or died mid-training (grace1 rebooted on 2026-10-22 and again on
# 2026-10-27). Resumes in the run's own folder from its last complete epoch snapshot, which carries
# the Adam state, so the shot stream and the schedule continue unchanged.
#   bash resume_student.sh kd9_10x70
# A run that has not reached SAVE_EPOCHS_FROM (epoch 40) has no snapshot and must be restarted.
#
# The last complete epoch is the largest number train_student.py printed, taken over every part
# log and the current run.log, not a count of lines: after a resume run.log starts again at the
# resume epoch, so counting its lines gives the wrong epoch. An epoch counts as complete only when
# its grad-norm line is followed by the Keras timing line, which it prints after the checkpoints.
set -uo pipefail
RUN=$1
HERE="$(cd "$(dirname "$0")" && pwd)"
DEPTH="$HERE/../depth_scaling_d7_2026-10"
HOSTPY=${HOSTPY:-$HOME/tfenv/bin/python}

OUT=$(ls -d "$HERE"/runs/${RUN}_2026* 2>/dev/null | tail -1)
[ -n "$OUT" ] || { echo "no run folder for $RUN"; exit 1; }
[ -f "$OUT/seed0/COMPLETE" ] && { echo "$OUT already COMPLETE"; exit 1; }
podman ps --format '{{.Names}}' | grep -qx "d9_$RUN" && { echo "d9_$RUN is still running"; exit 1; }

# rotate the current run.log into the next part, so this stretch's epochs are not overwritten
L="$OUT/seed0/run.log"
if [ -s "$L" ]; then
  n=1; while [ -e "$OUT/seed0/run_part$n.log" ]; do n=$((n + 1)); done
  prev=$((n - 1))
  if [ "$prev" -ge 1 ] && cmp -s "$L" "$OUT/seed0/run_part$prev.log"; then
    :                                  # paused and not yet resumed: part$prev is this same log
  else
    cp "$L" "$OUT/seed0/run_part$n.log"
  fi
fi
PARTS=("$OUT"/seed0/run_part*.log)
[ -e "${PARTS[0]}" ] || { echo "no part log in $OUT/seed0"; exit 1; }

DONE_EP=$("$HOSTPY" - "${PARTS[@]}" <<'PY'
import re, sys
done = set()
grad = re.compile(r'^\[gru\] epoch (\d+): global grad norm')
timing = re.compile(r'^\d+/\d+ - \d+s - loss')
for path in sys.argv[1:]:
    pending = None
    for line in open(path, errors='replace'):
        m = grad.match(line)
        if m:
            pending = int(m.group(1))
        elif pending is not None and timing.match(line):
            done.add(pending)
            pending = None
print(max(done) if done else 0)
PY
)
[ "${DONE_EP:-0}" -gt 0 ] || { echo "no complete epoch in $OUT/seed0 part logs"; exit 1; }
W=$(printf '%s/seed0/epochs/epoch%03d.weights.h5' "$OUT" "$DONE_EP")
[ -s "$W" ] || { echo "no epoch $DONE_EP snapshot at $W (epochs are saved from 40); restart instead"; exit 1; }
CUDA_VISIBLE_DEVICES= "$HOSTPY" -c "
import h5py, sys
with h5py.File(sys.argv[1]) as h:
    assert 'optimizer' in h, 'no optimizer state'
" "$W" || { echo "$W unreadable or without Adam state"; exit 1; }

LOGARGS=(); for f in "${PARTS[@]}"; do LOGARGS+=(--log "$f"); done
"$HOSTPY" "$DEPTH/history_from_log.py" "${LOGARGS[@]}" --epochs "$DONE_EP" \
  --out "$OUT/seed0/resume_history.json" || exit 1

rm -f "$OUT/PAUSED"
export RESUME_WEIGHTS="$W" INITIAL_EPOCH="$DONE_EP" RESUME_OUT="$OUT"
export RESUME_BEST_JSON=$(ls "$OUT"/seed0/ckpt/*.best.json 2>/dev/null | head -1)
export RESUME_HISTORY="$OUT/seed0/resume_history.json"
echo "$(date -u +%FT%TZ) $RUN resuming from epoch $DONE_EP in $(basename "$OUT")" >> "$HERE/status.log"
exec bash "$HERE/run_student.sh" "$RUN"
