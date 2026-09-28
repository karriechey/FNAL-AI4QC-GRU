#!/usr/bin/env bash
# Created: 2026-09-28
# Last updated: 2026-09-28
#
# Follow-up watcher for K2 (d7 u280 50M seed 1, cosine LR + global_clipnorm 0.64).
#
# Why: driver.sh launched K2 at 06:45:39Z and, in the same loop pass, checked whether the
# K2 container was running before podman had created it. It therefore logged
# "FAIL K2 ended without COMPLETE (exit )" and stopped watching K2, although K2 was
# starting normally. This script takes over what driver.sh would have done for K2: the
# clipping check (stop if the cumulative clipped fraction exceeds 0.05), the DONE line, the
# pairings K2 vs M2, K2 vs K1, K3 vs K2, and a re-draw of loss_curves.png with K2 included.
# It writes to driver_k2.log; driver.log is left as it was.
#
#     setsid nohup bash driver_k2.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail

REPO="$HOME/FNAL-AI4QC-GRU"
DIR="$REPO/experiments/cosine_2026-09-27"
MRN="$REPO/experiments/morning_2026-09-27"
LOG="$DIR/driver_k2.log"
IMG=qdec:tf215
PYH="$HOME/tfenv/bin/python"
CLIP_LIMIT=0.05
M2_DUMP=$(ls "$MRN"/M2_d7_u280_50M_s0_*/seed0/per_shot_*.npz)
NAME=K2_d7_u280_50M_s1
OUT=$(cat "$DIR/$NAME.out"); SD="$OUT/seed1"
S1=$(cat "$DIR/K1_d7_u280_50M_s0.out")/seed0
S3=$(cat "$DIR/K3_d7_u560_50M_s0.out")/seed0

log () { echo "[cosine-k2 $(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG"; }
running () { [ -n "$(podman ps -q --filter name="$1")" ]; }
eval_row () {
  python3 -c "
import csv,sys; r=list(csv.DictReader(open('$1/eval_best_ckpt.csv')))[-1]
print(r['p_L'], r['mwpm_p_L'], r['ratio'], r['student_only'], r['mwpm_only'], r['n_discordant'])"
}
best_epoch () { python3 -c "import json,glob;print(json.load(open(glob.glob('$1/ckpt/*.best.json')[0]))['best_epoch'])"; }
pair () {      # pair <a seeddir> <b dump> <a name> <b name>
  ( cd "$REPO/src" && "$PYH" pair_two_decoders.py --a "$(ls "$1"/per_shot_*.npz)" --b "$2" \
      --a-name "$3" --b-name "$4" --a-column student_correct --b-column student_correct \
      --out-csv "$1/mcnemar_vs_$4.csv" ) >> "$DIR/pairing.log" 2>&1 \
    && log "PAIR  $3 vs $4 -> $1/mcnemar_vs_$4.csv" || log "PAIR  $3 vs $4 FAILED"
}

log "BEGIN K2 watcher pid=$$ (driver.sh logged a false FAIL for K2 at 06:45:39Z; K2 container is running)"
running "cos_$NAME" || { log "K2 container not running at watcher start"; exit 1; }
while running "cos_$NAME"; do
  f=$(grep -o 'cumulative [0-9.]*' "$SD/run.log" 2>/dev/null | tail -1 | cut -d' ' -f2)
  if [ -n "$f" ] && python3 -c "import sys; sys.exit(0 if float('$f') > $CLIP_LIMIT else 1)"; then
    podman stop -t 30 "cos_$NAME" >/dev/null
    echo "cumulative clipped fraction $f > $CLIP_LIMIT" > "$OUT/STOP_clipping_over_5pct"
    log "STOP  $NAME: cumulative clipped fraction $f > $CLIP_LIMIT"; exit 0
  fi
  sleep 60
done
sleep 5
if [ ! -f "$SD/COMPLETE" ]; then log "FAIL  $NAME ended without COMPLETE (exit $(cat "$OUT/exit_code" 2>/dev/null))"; exit 0; fi
log "DONE  $NAME: eval p_L mwpm ratio b c n_disc = $(eval_row "$SD"); best epoch $(best_epoch "$SD"); $(grep -m1 -o 'params=[0-9,]*' "$SD/run.log"); clipped $(grep -o 'cumulative [0-9.]*' "$SD/run.log" | tail -1)"
pair "$SD" "$M2_DUMP" K2_d7_u280_50M_s1 M2_const_u280_50M_s0
pair "$SD" "$(ls "$S1"/per_shot_*.npz)" K2_d7_u280_50M_s1 K1_cos_u280_50M_s0
pair "$S3" "$(ls "$SD"/per_shot_*.npz)" K3_d7_u560_50M_s0 K2_cos_u280_50M_s1
podman run --rm --security-opt=label=disable -v "$HOME:$HOME" -e CUDA_VISIBLE_DEVICES= -w "$DIR" "$IMG" \
  python3 plot_curves.py > "$DIR/plot_curves.log" 2>&1 && log "PLOT  loss_curves.png (with K2)" || log "PLOT  failed (plot_curves.log)"
log "END   K2 finished"
