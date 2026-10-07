#!/usr/bin/env bash
# Created: 2026-10-07
# Last modified: 2026-10-07
#
# d9 first (2026-10-07 ~18:00Z): pause 9-9 (batch 5000, the least informative d9 run) to start
# 9-S16 (d9, 2 x u160, 9-1 / 9-S recipe) now; 9-S (d9, 2 x u280) keeps running.
#   1. 9-9 is supervised by the queue driver: the driver is stopped first (it would otherwise
#      record the stopped run as failed), then 9-9's container at its next epoch boundary;
#      PAUSED written. Its footprint before the pause is recorded.
#   2. 9-S16 starts when host memory available >= 9-S's measured footprint + 40 GB and nothing
#      is loading; its own footprint is measured after epoch 1.
#   3. 9-9 resumes (run_one.py --resume) when available >= its own footprint + 40 GB.
#   4. The queue driver is restarted (only 9-10, the replicate, is left in the queue).
#
#     setsid nohup bash d9_first.sh > d9_first.out 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
Q="$HOME/FNAL-AI4QC-GRU-mwpm/experiments/mwpm_fixes_d9_2026-10"
FOOT="$HERE/footprints.tsv"
PY="$HOME/tfenv/bin/python"
R02=$(ls $HOME/FNAL-AI4QC-GRU-compact/experiments/queue12_2026-09-30/R02_d9_u280_100M_2*/seed0/runs/*.history.json)
log () { echo "[probe $(date -u +%FT%TZ)] $*" | tee -a "$HERE/probe.log"; echo "[mwpm-d9 $(date -u +%FT%TZ)] NOTE  $*" >> "$Q/driver.log"; }
avail () { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
swap_mb () { free -m | awk '/Swap/{print $3}'; }
gb () { podman stats --no-stream --format '{{.MemUsage}}' "$1" 2>/dev/null | cut -d' ' -f1 | sed 's/GB//' | awk '{printf "%d", $1 + 0.5}'; }
foot_gb () { awk -F'\t' -v a="$1" '$2 == a {v = $4} END {sub(/GB/, "", v); printf "%d", v + 0.5}' "$FOOT"; }

# 1. pause 9-9
L="$(ls -d "$Q"/9-9_2* | tail -1)/seed0/run.log"
F99=$(gb md9_9-9)
n0=$(grep -c "global grad norm" "$L")
log "d9_first: pausing 9-9 at the end of its current epoch (footprint ${F99} GB) to start 9-S16 now"
until [ "$(grep -c "global grad norm" "$L")" -gt "$n0" ]; do sleep 2; done
for p in $(ps -eo pid,args | awk '$2 ~ /python/ && $3 == "driver.py" {print $1}'); do kill -9 "$p"; done
podman stop -t 30 md9_9-9 > /dev/null 2>&1
N=$(grep -c "global grad norm" "$L"); W=$(ls "$(dirname "$L")"/ckpt/*.lastepoch.weights.h5)
echo "paused after epoch $N at $(date -u +%FT%TZ); lastepoch weights $(date -u -r "$W" +%H:%M:%SZ); queue driver stopped first; footprint before the pause ${F99} GB" > "$(dirname "$(dirname "$L")")/PAUSED"
log "PAUSE 9-9 after epoch $N (queue driver stopped first); host available $(avail) GB"

# 2. 9-S16
NEED=$(( $(foot_gb 9s) + 40 ))
until [ "$(avail)" -ge "$NEED" ]; do sleep 120; done
log "starting 9s16 (d9 2 x u160; host available $(avail) GB >= $NEED = 9-S footprint + 40; swap $(swap_mb) MB)"
(cd "$HERE" && setsid nohup bash run_arm2.sh 9s16 > arm_9s16.out 2>&1 < /dev/null &)
sleep 90
until LS=$(ls "$HERE"/9s16_*/seed0/run.log 2>/dev/null) && grep -qE "\] epoch 1: val_loss|Traceback|STOP" "$LS"; do sleep 60; done
m=$(podman stats --no-stream --format '{{.MemUsage}}' probe_9s16 | cut -d' ' -f1)
echo -e "$(date -u +%FT%TZ)\t9s16\t$(grep -oE 'params=[0-9,]+' "$LS" | head -1)\t$m\t$(avail)\t$(swap_mb)" >> "$FOOT"
log "9s16 epoch 1 done: $(grep -E 'params=|\] epoch 1: val_loss|Traceback|STOP' "$LS" | cut -c1-120 | tr '\n' ' '); footprint $m; host available $(avail) GB"

# 3. resume 9-9
until [ "$(avail)" -ge $((F99 + 40)) ]; do sleep 300; done
log "resuming 9-9 (host available $(avail) GB >= $((F99 + 40)))"
OLD=$(ls -d "$Q"/9-9_2* | tail -1)
(cd "$Q" && setsid nohup "$PY" run_one.py 9-9 --resume --need-host-gb $((F99 + 40)) --baseline-history "$R02" > run_one_9-9_resume.stdout 2>&1 < /dev/null &)
sleep 120
NEW=$(ls -d "$Q"/9-9_2* | tail -1)
until grep -qE "\] epoch [0-9]+: val_loss|Traceback|STOP" "$NEW/seed0/run.log" 2>/dev/null; do sleep 60; done
log "9-9 resumed: train loss last 3 epochs before the pause $(grep -oE ' - loss: [0-9.]+' "$OLD/seed0/run.log" | tail -3 | awk '{print $3}' | tr '\n' ' '); first resumed epoch $(grep -oE ' - loss: [0-9.]+' "$NEW/seed0/run.log" | head -1 | awk '{print $3}'); host available $(avail) GB, swap $(swap_mb) MB"

# 4. queue driver back
(cd "$Q" && setsid nohup "$PY" driver.py --baseline-history "$R02" > driver.stdout 2>&1 < /dev/null &)
log "queue driver restarted"
