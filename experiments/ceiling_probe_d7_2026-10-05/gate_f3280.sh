#!/usr/bin/env bash
# Created: 2026-10-11
# Last modified: 2026-10-11
#
# Waits for the f3280 run (d9, 3 x u280, fresh shots, peak 5e-4) to finish and be scored on
# the evaluation block, then starts the follow-up arms of run_arm8.sh only if its best
# checkpoint is below MWPM there (ratio < 1):
#   f3280s1    replicate: initialisation seed 1, fresh-shot seed base 2*10**12
#   f3280e240  the same recipe for 240 epochs
#   ku280      one GRU layer of 280, distilled from the f3280 best checkpoint on fresh shots
#   ku100      one GRU layer of 100, the same
# Each arm starts only when the host has at least NEED_GB available (fresh-shot runs hold about
# 5-10 GB; NEED_GB keeps the box out of swap). Decisions are written to probe.log.
#
# f3280 runs from another code copy (RUN_DIR, default ~/FNAL-AI4QC-GRU-fresh): its launcher is
# being executed by bash and must not be overwritten, so the follow-up arms run from this copy.
#
#     setsid nohup bash gate_f3280.sh > gate_f3280.out 2>&1 < /dev/null &   (on grace1)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="${RUN_DIR:-$HOME/FNAL-AI4QC-GRU-fresh/experiments/ceiling_probe_d7_2026-10-05}"
NEED_GB="${NEED_GB:-60}"
log () { echo "[gate $(date -u +%FT%TZ)] $*" | tee -a "$HERE/probe.log"; }

OUT=$(ls -d "$RUN_DIR"/f3280_u280x280x280_* | sort | tail -1)
log "waiting for f3280 ($OUT) to be scored on the evaluation block"
until grep -q "trained arm f3280 exit" "$RUN_DIR/probe.log" && ls "$OUT"/seed0/eval*.csv >/dev/null 2>&1; do
  sleep 300
done
RATIO=$(python3 - "$OUT" <<'PY'
import csv, glob, sys
rows = [r for f in glob.glob(sys.argv[1] + '/seed0/eval*.csv') for r in csv.DictReader(open(f))]
print(rows[-1]['ratio'] if rows else '')
PY
)
log "f3280 evaluation block: ratio to MWPM $RATIO"
if [ -z "$RATIO" ] || ! python3 -c "import sys; sys.exit(0 if float('$RATIO') < 1.0 else 1)"; then
  log "f3280 is not below MWPM on the evaluation block; follow-up arms not started"
  exit 0
fi
for arm in f3280s1 f3280e240 ku280 ku100; do
  until [ "$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)" -ge "$NEED_GB" ]; do sleep 120; done
  log "f3280 below MWPM ($RATIO): starting $arm"
  setsid nohup bash "$HERE/run_arm8.sh" "$arm" > "$HERE/arm_$arm.out" 2>&1 < /dev/null &
  sleep 180          # let the arm load its validation and evaluation blocks before the next
done
log "all follow-up arms started"
