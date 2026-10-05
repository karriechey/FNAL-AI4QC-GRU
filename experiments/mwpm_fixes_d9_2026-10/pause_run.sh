#!/usr/bin/env bash
# Created: 2026-10-05
# Last modified: 2026-10-05
#
# Pause a queue run started by run_one.py at the next epoch boundary, to free host memory.
# Waits for the next "epoch N: global grad norm" line in the run log, which train_student.py
# prints after the epoch's checkpoints are written (.lastepoch.weights.h5 = model + optimizer
# state incl. the update count, which also fixes the LR-schedule step; .best.* and .best.json
# = best-so-far val p_L). Then stops run_one.py without letting it record an exit code (the
# queue driver keeps seeing the run as unfinished, so the promotion waits for it), stops the
# container, and writes PAUSED with the epoch. Resume: run_one.py <name> --resume.
#
#     bash pause_run.sh <name>
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
name="$1"
OUT=$(ls -d "$DIR/${name}"_2* | tail -1); L="$OUT/seed0/run.log"
n0=$(grep -c "global grad norm" "$L")
log () { echo "[mwpm-d9 $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$DIR/driver.log"; }
log "NOTE  pause requested for $name (memory for the d7 ceiling probe); waiting for the end of epoch $((n0 + 1))"
until [ "$(grep -c "global grad norm" "$L")" -gt "$n0" ]; do sleep 2; done
for p in $(pgrep -f "[r]un_one.py $name "); do kill -9 "$p"; done
podman stop -t 30 "md9_$name" > /dev/null
N=$(grep -c "global grad norm" "$L")
W=$(ls "$OUT"/seed0/ckpt/*.lastepoch.weights.h5)
echo "paused after epoch $N at $(date -u +%Y-%m-%dT%H:%M:%SZ); lastepoch weights $(date -u -r "$W" +%H:%M:%SZ)" > "$OUT/PAUSED"
log "PAUSE $name after epoch $N ($(cat "$OUT/PAUSED"))"
