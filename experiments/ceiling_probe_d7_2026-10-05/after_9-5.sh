#!/usr/bin/env bash
# Created: 2026-10-05
# Last modified: 2026-10-05
#
# When 9-5 ends: start ceiling-probe arm b (2 x u280) first, then resume 9-11 once arm b has
# finished loading and enough host memory is free (no swap). Then check that 9-11's first
# resumed epoch continues the training-loss trend of the epochs before the pause.
# Arm b runs from ~/FNAL-AI4QC-GRU-probe-b (a copy of the code with stacked-GRU support), so
# no file of a running job is changed. Everything is logged to probe.log and driver.log.
#
#     setsid nohup bash after_9-5.sh > after_9-5.out 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
Q="$HOME/FNAL-AI4QC-GRU-mwpm/experiments/mwpm_fixes_d9_2026-10"
B="$HOME/FNAL-AI4QC-GRU-probe-b/experiments/ceiling_probe_d7_2026-10-05"
R02=$(ls $HOME/FNAL-AI4QC-GRU-compact/experiments/queue12_2026-09-30/R02_d9_u280_100M_2*/seed0/runs/*.history.json)
NEED_GB=175
log () { echo "[probe $(date -u +%FT%TZ)] $*" | tee -a "$HERE/probe.log"; echo "[mwpm-d9 $(date -u +%FT%TZ)] NOTE  $*" >> "$Q/driver.log"; }
avail () { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
swap () { free -g | awk '/Swap/{print $3}'; }

log "after_9-5: waiting for 9-5 to end"
until grep -qE "END   9-5" "$Q/driver.log"; do sleep 60; done
sleep 30
log "9-5 ended; starting probe arm b (host available $(avail) GB)"
(cd "$B" && setsid nohup bash run_arm.sh b > arm_b.out 2>&1 < /dev/null &)
sleep 60
until L=$(ls "$B"/b_u280x280_*/seed0/run.log 2>/dev/null) && grep -qE "\] epoch 1: val_loss|Traceback|STOP" "$L"; do sleep 60; done
log "arm b loaded: $(grep -E 'params=|\] epoch 1: val_loss|Traceback|STOP' "$L" | tr '\n' ' ' | cut -c1-200); host available $(avail) GB, swap $(swap) GB"
until [ "$(avail)" -ge "$NEED_GB" ]; do log "9-11 resume waits: host available $(avail) GB < $NEED_GB"; sleep 600; done
log "resuming 9-11 (host available $(avail) GB)"
(cd "$Q" && setsid nohup "$HOME/tfenv/bin/python" run_one.py 9-11 --resume --baseline-history "$R02" > run_one_9-11_resume.stdout 2>&1 < /dev/null &)
sleep 120
NEW=$(ls -d "$Q"/9-11_2* | tail -1)
until grep -qE "\] epoch [0-9]+: val_loss|Traceback|STOP" "$NEW/seed0/run.log" 2>/dev/null; do sleep 60; done
OLD=$(ls -d "$Q"/9-11_2* | tail -2 | head -1)
before=$(grep -oE ' - loss: [0-9.]+' "$OLD/seed0/run.log" | tail -3 | awk '{print $3}' | tr '\n' ' ')
first=$(grep -oE ' - loss: [0-9.]+' "$NEW/seed0/run.log" | head -1 | awk '{print $3}')
log "9-11 resumed: $(grep -E 'resumed from|shuffle order advanced|optimizer update count' "$NEW/seed0/run.log" | head -2 | cut -c1-160 | tr '\n' ' '); train loss last 3 epochs before pause: $before; first resumed epoch: $first; host available $(avail) GB, swap $(swap) GB"
