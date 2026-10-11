#!/usr/bin/env bash
# Created: 2026-10-11
# Last modified: 2026-10-11
#
# Pause a probe arm (started by run_arm*.sh, container probe_<arm>) at the next epoch boundary,
# to give the GPU to other runs. Waits for the next "epoch N: global grad norm" line, which
# train_student.py prints after the epoch's checkpoints are written (.lastepoch.weights.h5 =
# model + optimizer state including the update count, so the LR schedule continues where it
# stopped; .best.* and .best.json = best so far by val p_L). Then stops the run_arm wrapper
# first (so it does not go on to score a half-trained checkpoint), stops the container, and
# writes PAUSED with the epoch. Fresh-shot runs resume on the seeds of the next epoch
# (fresh_shots.FreshShots.dataset, first_epoch = --initial-epoch).
#
#     bash pause_probe.sh <arm> [<folder>]      (on grace1, from the folder holding the arm)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
arm="$1"
OUT="${2:-$(ls -d "$HERE/${arm}"_u*Z | sort | tail -1)}"
L=$(ls "$OUT"/seed*/run.log)
log () { echo "[probe $(date -u +%FT%TZ)] $*" | tee -a "$HERE/probe.log"; }
n0=$(grep -c "global grad norm" "$L")
log "pause requested for $arm (GPU to the d9 runs); waiting for the end of epoch $((n0 + 1))"
until [ "$(grep -c "global grad norm" "$L")" -gt "$n0" ]; do sleep 5; done
for p in $(pgrep -f "[r]un_arm[0-9]*.sh $arm\$"); do kill "$p"; done
podman stop -t 30 "probe_$arm" > /dev/null
N=$(grep -c "global grad norm" "$L")
W=$(ls "$OUT"/seed*/ckpt/*.lastepoch.weights.h5)
echo "paused after epoch $N at $(date -u +%FT%TZ); lastepoch weights $(date -u -r "$W" +%FT%TZ)" > "$OUT/PAUSED"
log "PAUSE $arm after epoch $N ($OUT)"
