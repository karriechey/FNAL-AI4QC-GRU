#!/usr/bin/env bash
# Created: 2026-10-05
# Last modified: 2026-10-05
#
# Arm b first (requested 2026-10-05 13:58Z): pause 9-5 at its next epoch boundary, start
# ceiling-probe arm b (2 x u280), then resume the paused queue runs as host memory allows
# (no swap), then restart the queue driver. Replaces after_9-5.sh.
#   1. 9-5 is supervised by the queue driver itself, so the driver is stopped first (it would
#      otherwise record the stopped run as failed); then the container; PAUSED written.
#   2. arm b starts from ~/FNAL-AI4QC-GRU-probe-b (stacked-GRU code); wait until it has loaded.
#   3. Every 5 min: resume 9-5 if >= 300 GB is available (it holds ~261 GB), else 9-11 if
#      >= 175 GB (~135 GB). One resume at a time; the next decision waits for the previous
#      resume's first epoch. Each resume logs its first epoch's train loss next to the last
#      three before the pause.
#   4. When no queue run is paused any more, restart driver.py (same arguments).
#
#     setsid nohup bash arm_b_first.sh > arm_b_first.out 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
Q="$HOME/FNAL-AI4QC-GRU-mwpm/experiments/mwpm_fixes_d9_2026-10"
B="$HOME/FNAL-AI4QC-GRU-probe-b/experiments/ceiling_probe_d7_2026-10-05"
PY="$HOME/tfenv/bin/python"
R02=$(ls $HOME/FNAL-AI4QC-GRU-compact/experiments/queue12_2026-09-30/R02_d9_u280_100M_2*/seed0/runs/*.history.json)
log () { echo "[probe $(date -u +%FT%TZ)] $*" | tee -a "$HERE/probe.log"; echo "[mwpm-d9 $(date -u +%FT%TZ)] NOTE  $*" >> "$Q/driver.log"; }
avail () { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
swap () { free -g | awk '/Swap/{print $3}'; }
latest () { ls -d "$Q/$1"_2* | tail -1; }
paused () { [ -f "$(latest "$1")/PAUSED" ]; }

for p in $(pgrep -f "[a]fter_9-5.sh"); do kill "$p"; done
log "arm_b_first: after_9-5.sh stopped; pausing 9-5 at the end of its current epoch"

# 1. pause 9-5
L="$(latest 9-5)/seed0/run.log"; n0=$(grep -c "global grad norm" "$L")
until [ "$(grep -c "global grad norm" "$L")" -gt "$n0" ]; do sleep 2; done
for p in $(pgrep -f "[d]river.py --baseline"); do kill -9 "$p"; done
podman stop -t 30 md9_9-5 > /dev/null 2>&1
N=$(grep -c "global grad norm" "$L"); W=$(ls "$(latest 9-5)"/seed0/ckpt/*.lastepoch.weights.h5)
echo "paused after epoch $N at $(date -u +%FT%TZ); lastepoch weights $(date -u -r "$W" +%H:%M:%SZ); queue driver stopped first" > "$(latest 9-5)/PAUSED"
log "PAUSE 9-5 after epoch $N (queue driver stopped first so the run is not recorded as failed); host available $(avail) GB"

# 2. arm b
(cd "$B" && setsid nohup bash run_arm.sh b > arm_b.out 2>&1 < /dev/null &)
sleep 60
until LB=$(ls "$B"/b_u280x280_*/seed0/run.log 2>/dev/null) && grep -qE "\] epoch 1: val_loss|Traceback|STOP" "$LB"; do sleep 60; done
log "arm b loaded: $(grep -E 'params=|\] epoch 1: val_loss|Traceback|STOP' "$LB" | tr '\n' ' ' | cut -c1-220); host available $(avail) GB, swap $(swap) GB"

# 3. resumes
resume () {  # resume <name>
  local name="$1" OLD NEW before first
  OLD=$(latest "$name")
  log "resuming $name (host available $(avail) GB)"
  (cd "$Q" && setsid nohup "$PY" run_one.py "$name" --resume --baseline-history "$R02" > "run_one_${name}_resume.stdout" 2>&1 < /dev/null &)
  sleep 120
  NEW=$(latest "$name")
  until grep -qE "\] epoch [0-9]+: val_loss|Traceback|STOP" "$NEW/seed0/run.log" 2>/dev/null; do sleep 60; done
  before=$(grep -oE ' - loss: [0-9.]+' "$OLD/seed0/run.log" | tail -3 | awk '{print $3}' | tr '\n' ' ')
  first=$(grep -oE ' - loss: [0-9.]+' "$NEW/seed0/run.log" | head -1 | awk '{print $3}')
  log "$name resumed: $(grep -E 'resumed from|shuffle order advanced' "$NEW/seed0/run.log" | cut -c1-130 | tr '\n' ' '); train loss last 3 epochs before pause: $before; first resumed epoch: $first; host available $(avail) GB, swap $(swap) GB"
}
while paused 9-5 || paused 9-11; do
  if paused 9-5 && [ "$(avail)" -ge 300 ]; then resume 9-5
  elif paused 9-11 && [ "$(avail)" -ge 175 ]; then resume 9-11
  else sleep 300; fi
done

# 4. queue driver back
(cd "$Q" && setsid nohup "$PY" driver.py --baseline-history "$R02" > driver.stdout 2>&1 < /dev/null &)
log "queue driver restarted (no queue run paused any more)"
