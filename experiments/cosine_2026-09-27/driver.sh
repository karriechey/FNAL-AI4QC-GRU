#!/usr/bin/env bash
# Created: 2026-09-27
# Last updated: 2026-09-27
#
# Cosine-schedule queue 2026-09-27 (pre-registered before any result), run unattended on
# grace1. Fix for the late rise in training loss of the 50M runs at a constant 3e-3.
#
# Schedule for every job: 80 epochs, cosine LR 3e-3 -> 3e-5 per update (no warmup), Adam
# global_clipnorm 0.64 (= 5 x median global gradient norm of updates 100-500 of the probe in
# ../gclip_probe_2026-09-27, 2 significant figures). Everything else as before:
# reset_after=False, batch 10,000, early stopping off, determinism on, TF32 on for training /
# off for evaluation, best checkpoint by validation LER, evaluation block and MWPM on the
# same shots, McNemar. 1.6 GB GPU cap; a job starts only with >= 2 GB GPU memory free.
#
#   K1  d7 u280 50M seed 0   (compare with M2: same data and seed, constant LR, p_L 0.006233)
#   K3  d7 u560 50M seed 0   (1,023,681 parameters)
#   K2  d7 u280 50M seed 1   (starts when K1 has finished)
#   50M = FORMAL 15M + TRAINONLY seed43 5M + seed44 30M.
#
# Pre-registered questions: (a) does the schedule remove the late-training rise in training
# loss? (b) does u280 50M improve on 1.45x MWPM? (c) does u560 improve on u280 at 50M?
#
# Clipping check: after every epoch each run prints the cumulative fraction of clipped
# updates. If it exceeds 0.05 the run is stopped, STOP_clipping_over_5pct is written, and no
# further job is started.
#
#     setsid nohup bash driver.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail

REPO="$HOME/FNAL-AI4QC-GRU"
DIR="$REPO/experiments/cosine_2026-09-27"
MRN="$REPO/experiments/morning_2026-09-27"
LOG="$DIR/driver.log"
IMG=qdec:tf215
PYH="$HOME/tfenv/bin/python"
CAP=1600
NEED_FREE=2000
GCLIP=0.64
CLIP_LIMIT=0.05
M2_DUMP=$(ls "$MRN"/M2_d7_u280_50M_s0_*/seed0/per_shot_*.npz)
P7="$HOME/pools_d7_p004_19M/data_d7_p0.004_r7"
X50="${P7}_TRAINONLY_seed43.npz ${P7}_TRAINONLY_seed44.npz"
N50="5000000 30000000"
mkdir -p "$DIR"

log () { echo "[cosine $(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG"; }
free_mib () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
running () { [ -n "$(podman ps -q --filter name="$1")" ]; }

launch () {   # launch <name> <units> <seed>
  local name="$1" U="$2" SEED="$3" S OUT
  S=$(date -u +%Y%m%dT%H%M%SZ); OUT="$DIR/${name}_$S"; mkdir -p "$OUT"
  { echo "at_start repo_head $(git -C "$REPO" rev-parse HEAD)"
    echo "at_start train_student_sha256 $(sha256sum "$REPO/src/train_student.py" | cut -d' ' -f1)"
  } > "$OUT/code_commit.txt"
  echo "$OUT" > "$DIR/$name.out"
  log "START $name d=7 units=$U seed=$SEED 50M cosine 3e-3->3e-5 80 epochs global_clipnorm=$GCLIP -> $OUT (free GPU $(free_mib) MiB)"
  (podman run --rm --name "cos_$name" --security-opt=label=disable --device nvidia.com/gpu=all \
     -v "$HOME:$HOME" -e HOME="$HOME" -w "$REPO/src" "$IMG" \
     env D=7 ROUNDS=7 SEEDS="$SEED" UNITS="$U" NTRAIN=15000000 EXTRA_POOL="$X50" EXTRA_N="$N50" \
         BATCH=10000 EPOCHS=80 LR=0.003 LR_SCHEDULE=cosine LR_END=3e-5 GLOBAL_CLIPNORM="$GCLIP" \
         EARLY_STOP=0 RESET_AFTER=0 UNROLL=0 EVAL_NO_TF32=1 \
         GPU_MEM_MIB="$CAP" MIN_FREE_MIB="$CAP" OUT="$OUT" STAMP="$S" \
     bash gh200_run_student_scaling.sh > "$OUT/podman.log" 2>&1; echo $? > "$OUT/exit_code") &
}

seeddir () { echo "$(cat "$DIR/$1.out")/seed$2"; }
eval_row () {  # eval_row <seeddir> -> "p_L mwpm ratio b c nd"
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
clip_check () {  # clip_check <name> <seed>: stop the run if cumulative clipped fraction > limit
  local SD f
  SD=$(seeddir "$1" "$2"); [ -f "$SD/run.log" ] || return 0
  f=$(grep -o 'cumulative [0-9.]*' "$SD/run.log" | tail -1 | cut -d' ' -f2)
  [ -z "$f" ] && return 0
  if python3 -c "import sys; sys.exit(0 if float('$f') > $CLIP_LIMIT else 1)"; then
    podman stop -t 30 "cos_$1" >/dev/null
    echo "cumulative clipped fraction $f > $CLIP_LIMIT" > "$(cat "$DIR/$1.out")/STOP_clipping_over_5pct"
    log "STOP  $1: cumulative clipped fraction $f > $CLIP_LIMIT"
    STOPPED=yes
  fi
}
finish () {    # finish <name> <seed>
  local SD; SD=$(seeddir "$1" "$2")
  if [ ! -f "$SD/COMPLETE" ]; then log "FAIL  $1 ended without COMPLETE (exit $(cat "$(cat "$DIR/$1.out")/exit_code" 2>/dev/null))"; return; fi
  log "DONE  $1: eval p_L mwpm ratio b c n_disc = $(eval_row "$SD"); best epoch $(best_epoch "$SD"); $(grep -m1 -o 'params=[0-9,]*' "$SD/run.log"); clipped $(grep -o 'cumulative [0-9.]*' "$SD/run.log" | tail -1)"
}

log "BEGIN cosine driver pid=$$ repo=$(git -C "$REPO" rev-parse --short HEAD) global_clipnorm=$GCLIP"
STOPPED=""
until [ "$(free_mib)" -ge "$NEED_FREE" ]; do sleep 60; done
launch K1_d7_u280_50M_s0 280 0
sleep 300          # let the job allocate before measuring free memory again
until [ "$(free_mib)" -ge "$NEED_FREE" ]; do sleep 60; done
launch K3_d7_u560_50M_s0 560 0

K1=""; K2=""; K3=""; K2_STARTED=""
while :; do
  running cos_K1_d7_u280_50M_s0 && clip_check K1_d7_u280_50M_s0 0
  running cos_K3_d7_u560_50M_s0 && clip_check K3_d7_u560_50M_s0 0
  [ -n "$K2_STARTED" ] && running cos_K2_d7_u280_50M_s1 && clip_check K2_d7_u280_50M_s1 1
  if [ -z "$K1" ] && ! running cos_K1_d7_u280_50M_s0; then K1=yes; finish K1_d7_u280_50M_s0 0; fi
  if [ -z "$K3" ] && ! running cos_K3_d7_u560_50M_s0; then K3=yes; finish K3_d7_u560_50M_s0 0; fi
  if [ -n "$K1" ] && [ -z "$K2_STARTED" ]; then
    if [ -n "$STOPPED" ]; then
      log "NOT STARTING K2: a run was stopped for clipping"; K2_STARTED=no; K2=yes
    elif [ "$(free_mib)" -ge "$NEED_FREE" ]; then
      launch K2_d7_u280_50M_s1 280 1; K2_STARTED=yes
    fi
  fi
  if [ "$K2_STARTED" = yes ] && [ -z "$K2" ] && ! running cos_K2_d7_u280_50M_s1; then K2=yes; finish K2_d7_u280_50M_s1 1; fi
  [ -n "$K1" ] && [ -n "$K2" ] && [ -n "$K3" ] && break
  sleep 60
done

# Pairings (only between completed runs)
S1=$(seeddir K1_d7_u280_50M_s0 0); S3=$(seeddir K3_d7_u560_50M_s0 0)
S2=""; [ "$K2_STARTED" = yes ] && S2=$(seeddir K2_d7_u280_50M_s1 1)
[ -f "$S1/COMPLETE" ] && pair "$S1" "$M2_DUMP" K1_d7_u280_50M_s0 M2_const_u280_50M_s0
[ -n "$S2" ] && [ -f "$S2/COMPLETE" ] && pair "$S2" "$M2_DUMP" K2_d7_u280_50M_s1 M2_const_u280_50M_s0
[ -n "$S2" ] && [ -f "$S2/COMPLETE" ] && [ -f "$S1/COMPLETE" ] && pair "$S2" "$(ls "$S1"/per_shot_*.npz)" K2_d7_u280_50M_s1 K1_cos_u280_50M_s0
[ -f "$S3/COMPLETE" ] && [ -f "$S1/COMPLETE" ] && pair "$S3" "$(ls "$S1"/per_shot_*.npz)" K3_d7_u560_50M_s0 K1_cos_u280_50M_s0
[ -n "$S2" ] && [ -f "$S3/COMPLETE" ] && [ -f "$S2/COMPLETE" ] && pair "$S3" "$(ls "$S2"/per_shot_*.npz)" K3_d7_u560_50M_s0 K2_cos_u280_50M_s1
[ -f "$S3/COMPLETE" ] && pair "$S3" "$M2_DUMP" K3_d7_u560_50M_s0 M2_const_u280_50M_s0

podman run --rm --security-opt=label=disable -v "$HOME:$HOME" -e CUDA_VISIBLE_DEVICES= -w "$DIR" "$IMG" \
  python3 plot_curves.py > "$DIR/plot_curves.log" 2>&1 && log "PLOT  loss_curves.png" || log "PLOT  failed (plot_curves.log)"
log "END   cosine queue finished"
