#!/usr/bin/env bash
# Created: 2026-10-07
# Last modified: 2026-10-07
#
# d9 stacked queue, version 2 (2026-10-07): 9-9 is stopped for good (decision 2026-10-07, not a
# pre-registered rule: batch 5000 trailed 9-1 by ~50% at the same epoch and the d9 question is
# now depth). After 9-S16 has started, take over from d9_first.sh (stopped, so it never resumes
# 9-9), mark 9-9 stopped (KILLED file, results row), then run in order:
#   9-S3_160: d9, 3 x u160      9-S3_280: d9, 3 x u280
# both with the 9-1 / 9-S recipe; then restart the queue driver (only 9-10 is left).
# Each starts when host memory available >= 9-S's measured footprint + 40 GB and nothing is
# loading; each new architecture is measured after its first epoch before the next starts.
#
#     setsid nohup bash d9_queue.sh > d9_queue.out 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
Q="$HOME/FNAL-AI4QC-GRU-mwpm/experiments/mwpm_fixes_d9_2026-10"
FOOT="$HERE/footprints.tsv"
PY="$HOME/tfenv/bin/python"
R02=$(ls $HOME/FNAL-AI4QC-GRU-compact/experiments/queue12_2026-09-30/R02_d9_u280_100M_2*/seed0/runs/*.history.json)
log () { echo "[probe $(date -u +%FT%TZ)] $*" | tee -a "$HERE/probe.log"; echo "[mwpm-d9 $(date -u +%FT%TZ)] NOTE  $*" >> "$Q/driver.log"; }
avail () { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
swap_mb () { free -m | awk '/Swap/{print $3}'; }
foot_gb () { awk -F'\t' -v a="$1" '$2 == a {v = $4} END {sub(/GB/, "", v); printf "%d", v + 0.5}' "$FOOT"; }
loading () {
  local f
  for f in $(ls "$HERE"/*_2*/seed0/run.log "$Q"/9-*_2*/seed0/run.log 2>/dev/null); do
    [ -n "$(find "$f" -mmin -30)" ] && ! grep -q "\] epoch [0-9]*: val_loss" "$f" && return 0
  done
  return 1
}
start () {  # start <arm> <need_gb>
  until [ "$(avail)" -ge "$2" ] && ! loading; do sleep 120; done
  log "starting $1 (host available $(avail) GB >= $2; swap $(swap_mb) MB)"
  (cd "$HERE" && setsid nohup bash run_arm2.sh "$1" > "arm_$1.out" 2>&1 < /dev/null &)
  sleep 90
  local L m
  until L=$(ls "$HERE"/$1_*/seed0/run.log 2>/dev/null) && grep -qE "\] epoch 1: val_loss|Traceback|STOP" "$L"; do sleep 60; done
  m=$(podman stats --no-stream --format '{{.MemUsage}}' "probe_$1" | cut -d' ' -f1)
  echo -e "$(date -u +%FT%TZ)\t$1\t$(grep -oE 'params=[0-9,]+' "$L" | head -1)\t$m\t$(avail)\t$(swap_mb)" >> "$FOOT"
  log "$1 epoch 1 done: $(grep -E 'params=|\] epoch 1: val_loss|Traceback|STOP' "$L" | cut -c1-120 | tr '\n' ' '); footprint $m; host available $(avail) GB"
}

until grep -q "9s16 epoch 1 done" "$HERE/probe.log"; do sleep 60; done
for p in $(ps -eo pid,args | awk '$2 == "bash" && $3 == "d9_first.sh" {print $1}'); do kill "$p"; done
D99=$(ls -d "$Q"/9-9_2* | tail -1)
N99=$(sed -nE 's/^paused after epoch ([0-9]+).*/\1/p' "$D99/PAUSED")
echo "Killed @ ep ${N99}: stopped by decision 2026-10-07 (not a pre-registered rule): batch 5000 trailed 9-1 by about 50% in val p_L at the same epoch, and the d9 queue now tests depth; GPU time and memory given to the stacked d9 runs" > "$D99/KILLED"
"$PY" - "$D99" "$Q/results.tsv" "$N99" <<'PYROW'
import csv, json, glob, sys
d, res, n = sys.argv[1], sys.argv[2], sys.argv[3]
b = json.load(open(glob.glob(d + '/seed0/ckpt/*.best.json')[0]))
row = [f'Killed @ ep {n}', 'u280', '303521', 9, 9, 'cosine to 3e-5', '0.001', '100M', '0', 0.004, '5000', '160',
       b['best_epoch'], '', '', '', round(b['val_loss'], 5), 'FALSE',
       f"min val p_L; warmup 3 ep; clipnorm 0.64; queue 9-9 (batch 5000 at the promoted peak LR); stopped at epoch {n} by decision 2026-10-07, not scored; best val p_L {b['val_ler']} at epoch {b['best_epoch']}",
       '', '2026-10-07']
csv.writer(open(res, 'a', newline=''), delimiter='\t').writerow(row)
PYROW
log "d9_queue: d9_first.sh stopped after starting 9-S16; 9-9 stopped for good at epoch ${N99} (KILLED file, results row); next 3 x u160 and 3 x u280 at d9, then the queue driver restarts"
NEED=$(( $(foot_gb 9s) + 40 ))
start 9s316 "$NEED"
start 9s3280 "$NEED"
(cd "$Q" && setsid nohup "$PY" driver.py --baseline-history "$R02" > driver.stdout 2>&1 < /dev/null &)
log "queue driver restarted"
