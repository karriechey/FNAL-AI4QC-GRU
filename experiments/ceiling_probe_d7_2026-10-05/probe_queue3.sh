#!/usr/bin/env bash
# Created: 2026-10-06
# Last modified: 2026-10-06
#
# Order from 2026-10-06 ~19:00Z: 9-S, then S2, then 9-9. Takes over from probe_queue2.sh.
#   - D2 (already started): wait for its first epoch, measure.
#   - 9-S (2 x u280, d9, 9-1 recipe, no EMA): starts when >= 230 GB is available and nothing
#     is loading (expected when 9-7 ends, ~22:30Z); footprint measured after epoch 1.
#   - S2: when >= S1's measured footprint + 40 GB; measured after epoch 1.
#   - HOLD (keeps the queue's 9-9 back) is removed after S2 has started.
#   - 9-S+EMA follow-up, decided once 9-7 is scored, by the rule fixed here in advance:
#     queued only if 9-7's evaluation-block ratio is below 9-1's AND a paired McNemar test
#     9-7 vs 9-1 on the same 1.8M evaluation shots (pair_two_decoders.py on their per-shot
#     dumps) favours 9-7 with p < 0.01. If queued, it starts after 9-9 has loaded, when
#     >= 9-S's measured footprint + 40 GB is available. 9-S itself is not changed.
#
#     setsid nohup bash probe_queue3.sh > probe_queue3.out 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CD="$HOME/FNAL-AI4QC-GRU-probe-cd/experiments/ceiling_probe_d7_2026-10-05"
PB="$HOME/FNAL-AI4QC-GRU-probe-b/experiments/ceiling_probe_d7_2026-10-05"
Q="$HOME/FNAL-AI4QC-GRU-mwpm/experiments/mwpm_fixes_d9_2026-10"
FOOT="$HERE/footprints.tsv"
MARGIN=40
log () { echo "[probe $(date -u +%FT%TZ)] $*" | tee -a "$HERE/probe.log"; echo "[mwpm-d9 $(date -u +%FT%TZ)] NOTE  $*" >> "$Q/driver.log"; }
avail () { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
swap_mb () { free -m | awk '/Swap/{print $3}'; }
loading () {
  local f
  for f in $(ls "$HERE"/*_2*/seed0/run.log "$CD"/*_2*/seed0/run.log "$PB"/*_2*/seed0/run.log "$Q"/9-*_2*/seed0/run.log 2>/dev/null); do
    [ -n "$(find "$f" -mmin -30)" ] && ! grep -q "\] epoch [0-9]*: val_loss" "$f" && return 0
  done
  return 1
}
foot_gb () { awk -F'\t' -v a="$1" '$2 == a {v = $4} END {sub(/GB/, "", v); printf "%d", v + 0.5}' "$FOOT"; }
hold () { echo "2026-10-06: $1" > "$Q/HOLD"; }
measure () {
  local L m
  until L=$(ls "$HERE"/$1_*/seed0/run.log 2>/dev/null) && grep -qE "\] epoch 1: val_loss|Traceback|STOP" "$L"; do sleep 60; done
  m=$(podman stats --no-stream --format '{{.MemUsage}}' "probe_$1" 2>/dev/null | cut -d' ' -f1)
  echo -e "$(date -u +%FT%TZ)\t$1\t$(grep -oE 'params=[0-9,]+' "$L" | head -1)\t$m\t$(avail)\t$(swap_mb)" >> "$FOOT"
  log "$1 epoch 1 done: $(grep -E 'params=|teacher .*logits for all|\] epoch 1: val_loss|Traceback|STOP' "$L" | cut -c1-140 | tr '\n' ' '); footprint $m; host available $(avail) GB, swap $(swap_mb) MB"
}
start () {  # start <arm> <need_gb> <why> [run script]
  until [ "$(avail)" -ge "$2" ] && ! loading; do sleep 120; done
  log "starting $1 (host available $(avail) GB >= $2 = $3; swap $(swap_mb) MB)"
  (cd "$HERE" && setsid nohup bash "${4:-run_arm.sh}" "$1" > "arm_$1.out" 2>&1 < /dev/null &)
  sleep 90
  measure "$1"
}

for p in $(ps -eo pid,args | awk '$2 == "bash" && $3 == "probe_queue2.sh" {print $1}'); do kill "$p"; done
hold "order 9-S, S2, then 9-9"
log "probe_queue3: probe_queue2.sh stopped; order now 9-S (>= 230 GB), S2 (S1 + ${MARGIN} GB), then 9-9"
grep -qP '\td2\t' "$FOOT" || measure d2
start 9s 230 "230 GB rule (new architecture at d9)"
S1=$(foot_gb s1)
start s2 $((S1 + MARGIN)) "S1 footprint ${S1} GB + ${MARGIN}"
rm -f "$Q/HOLD"
log "9-S and S2 running; HOLD removed so the queue can start 9-9"

# 9-S+EMA decision (rule above)
until grep -qE "END   9-7" "$Q/driver.log"; do sleep 300; done
A=$(ls "$Q"/9-1_2*/seed0/per_shot_*.npz | head -1); B=$(ls "$(ls -d "$Q"/9-7_2* | tail -1)"/seed0/per_shot_*.npz | head -1)
P="$HERE/pair_9-1_vs_9-7.csv"; rm -f "$P"
(cd "$HOME/FNAL-AI4QC-GRU-probe-q/src" && "$HOME/tfenv/bin/python" pair_two_decoders.py --a "$A" --a-name 9-1 --b "$B" --b-name 9-7 --out-csv "$P" > "$HERE/pair_9-1_vs_9-7.log" 2>&1)
read -r pa pb bonly aonly ndisc pex <<< "$("$HOME/tfenv/bin/python" -c "
import csv; r = next(csv.DictReader(open('$P')))
print(r['p_L_a'], r['p_L_b'], r['b_only'], r['a_only'], r['n_discordant'], r['p_exact'])")"
if "$HOME/tfenv/bin/python" -c "import sys; sys.exit(0 if float('$pb') < float('$pa') and int('$bonly') > int('$aonly') and float('$pex') < 0.01 else 1)"; then
  log "9-7 vs 9-1 on the evaluation block: p_L ${pb} vs ${pa}, 9-7 only right ${bonly}, 9-1 only right ${aonly}, n_disc ${ndisc}, p ${pex}: 9-7 clearly better -> 9-S+EMA queued after 9-9"
  until L=$(ls "$Q"/9-9_2*/seed0/run.log 2>/dev/null) && grep -q "\] epoch [0-9]*: val_loss" "$L"; do sleep 300; done
  S9=$(foot_gb 9s)
  hold "9-S+EMA loading"
  start 9se $((S9 + MARGIN)) "9-S footprint ${S9} GB + ${MARGIN}" run_arm2.sh
  rm -f "$Q/HOLD"
  log "9-S+EMA running; HOLD removed"
else
  log "9-7 vs 9-1 on the evaluation block: p_L ${pb} vs ${pa}, 9-7 only right ${bonly}, 9-1 only right ${aonly}, n_disc ${ndisc}, p ${pex}: not clearly better -> no 9-S+EMA follow-up"
fi
