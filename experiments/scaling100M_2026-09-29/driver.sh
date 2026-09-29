#!/usr/bin/env bash
# Created: 2026-09-29
# Last updated: 2026-09-29
#
# Data-scaling run 2026-09-29 (pre-registered before any result), run unattended on grace1.
# Same recipe as K1/K2 in ../cosine_2026-09-27 (80 epochs, cosine LR 3e-3 -> 3e-5 per update,
# no warmup, Adam global_clipnorm 0.64, reset_after=False, batch 10,000, early stopping off,
# determinism on, TF32 on for training / off for evaluation, best checkpoint by validation
# LER, evaluation block [15.2M, 17M) and MWPM on the same shots, McNemar). 1.6 GB GPU cap.
#
#   L1  d7 u280 100M seed 0  (compare with K1: same recipe and seed at 50M, p_L 0.005364)
#       100M = FORMAL 15M + TRAINONLY seed43 5M + seed44 30M + seed45 50M (nested: the first
#       50M are exactly K1's training shots, in the same order).
#       Note: 80 epochs of 100M = 10,000 updates per epoch, 800,000 updates in total
#       (K1: 400,000); the cosine decay is laid out over all 800,000.
#   L2  d5 u140 20M seed 0   (compare with the d5 constant-LR control, p_L 0.007106)
#       20M = FORMAL 15M + TRAINONLY seed43 5M. Started only after L1 is training and only
#       if at least 2 GB GPU memory and 150 GB host memory are still free.
#
# Pre-registered questions: (primary) does 100M improve on 50M for d7 u280 under the same
# recipe? (secondary) does the cosine recipe keep or improve the d5 control?
# The sealed partition [17M, 19M) is not read.
#
# Clipping check: if a run's cumulative fraction of clipped updates exceeds 0.05 it is
# stopped and STOP_clipping_over_5pct is written.
# Memory: every 30 s, resources.tsv gets host memory available, each container's memory
# use (podman stats) and each process's GPU memory (nvidia-smi).
#
#     setsid nohup bash driver.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail

REPO="$HOME/FNAL-AI4QC-GRU"
DIR="$REPO/experiments/scaling100M_2026-09-29"
LOG="$DIR/driver.log"
RES="$DIR/resources.tsv"
IMG=qdec:tf215
PYH="$HOME/tfenv/bin/python"
CAP=1600
NEED_FREE=2000
NEED_HOST_GB=150
GCLIP=0.64
CLIP_LIMIT=0.05
K1_DIR=$(ls -d "$REPO"/experiments/cosine_2026-09-27/K1_d7_u280_50M_s0_*/seed0)
K2_DIR=$(ls -d "$REPO"/experiments/cosine_2026-09-27/K2_d7_u280_50M_s1_*/seed1)
CTL_DIR=$(ls -d "$REPO"/experiments/control_ra0/d5_s0_*/seed0)
P7="$HOME/pools_d7_p004_19M/data_d7_p0.004_r7"
P5="$HOME/pools_d5_p004_19M/data_d5_p0.004_r5"
mkdir -p "$DIR"

log () { echo "[scaling100M $(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG"; }
free_mib () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
host_avail_gb () { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
running () { [ -n "$(podman ps -q --filter name="$1")" ]; }

launch () {   # launch <name> <d> <units> <seed> <extra pools> <extra counts>
  local name="$1" D="$2" U="$3" SEED="$4" XP="$5" XN="$6" S OUT
  S=$(date -u +%Y%m%dT%H%M%SZ); OUT="$DIR/${name}_$S"; mkdir -p "$OUT"
  { echo "at_start repo_head $(git -C "$REPO" rev-parse HEAD)"
    echo "at_start train_student_sha256 $(sha256sum "$REPO/src/train_student.py" | cut -d' ' -f1)"
  } > "$OUT/code_commit.txt"
  echo "$OUT" > "$DIR/$name.out"
  log "START $name d=$D units=$U seed=$SEED extra=[$XN] cosine 3e-3->3e-5 80 epochs global_clipnorm=$GCLIP -> $OUT (free GPU $(free_mib) MiB, host available $(host_avail_gb) GB)"
  (podman run --rm --name "sc_$name" --security-opt=label=disable --device nvidia.com/gpu=all \
     -v "$HOME:$HOME" -e HOME="$HOME" -w "$REPO/src" "$IMG" \
     env D="$D" ROUNDS="$D" SEEDS="$SEED" UNITS="$U" NTRAIN=15000000 EXTRA_POOL="$XP" EXTRA_N="$XN" \
         BATCH=10000 EPOCHS=80 LR=0.003 LR_SCHEDULE=cosine LR_END=3e-5 GLOBAL_CLIPNORM="$GCLIP" \
         EARLY_STOP=0 RESET_AFTER=0 UNROLL=0 EVAL_NO_TF32=1 \
         GPU_MEM_MIB="$CAP" MIN_FREE_MIB="$CAP" OUT="$OUT" STAMP="$S" \
     bash gh200_run_student_scaling.sh > "$OUT/podman.log" 2>&1; echo $? > "$OUT/exit_code") &
  # Wait for the container to exist before anyone asks whether it is running (the
  # 2026-09-28 driver checked too early and logged a false FAIL).
  local i
  for i in $(seq 1 60); do running "sc_$name" && return 0; sleep 5; done
  log "WARN  $name container not seen 5 min after launch"
}

seeddir () { echo "$(cat "$DIR/$1.out")/seed$2"; }
eval_row () {
  python3 -c "
import csv; r=list(csv.DictReader(open('$1/eval_best_ckpt.csv')))[-1]
print(r['p_L'], r['mwpm_p_L'], r['ratio'], r['student_only'], r['mwpm_only'], r['n_discordant'])"
}
best_epoch () { python3 -c "import json,glob;print(json.load(open(glob.glob('$1/ckpt/*.best.json')[0]))['best_epoch'])"; }
pair () {      # pair <a seeddir> <b dump> <a name> <b name>
  ( cd "$REPO/src" && "$PYH" pair_two_decoders.py --a "$(ls "$1"/per_shot_*.npz)" --b "$2" \
      --a-name "$3" --b-name "$4" --a-column student_correct --b-column student_correct \
      --out-csv "$1/mcnemar_vs_$4.csv" ) >> "$DIR/pairing.log" 2>&1 \
    && log "PAIR  $3 vs $4 -> $1/mcnemar_vs_$4.csv" || log "PAIR  $3 vs $4 FAILED"
}
clip_check () {  # clip_check <name> <seed>
  local SD f
  SD=$(seeddir "$1" "$2"); [ -f "$SD/run.log" ] || return 0
  f=$(grep -o 'cumulative [0-9.]*' "$SD/run.log" | tail -1 | cut -d' ' -f2)
  [ -z "$f" ] && return 0
  if python3 -c "import sys; sys.exit(0 if float('$f') > $CLIP_LIMIT else 1)"; then
    podman stop -t 30 "sc_$1" >/dev/null
    echo "cumulative clipped fraction $f > $CLIP_LIMIT" > "$(cat "$DIR/$1.out")/STOP_clipping_over_5pct"
    log "STOP  $1: cumulative clipped fraction $f > $CLIP_LIMIT"
  fi
}
finish () {    # finish <name> <seed>
  local SD; SD=$(seeddir "$1" "$2")
  if [ ! -f "$SD/COMPLETE" ]; then log "FAIL  $1 ended without COMPLETE (exit $(cat "$(cat "$DIR/$1.out")/exit_code" 2>/dev/null))"; return 1; fi
  log "DONE  $1: eval p_L mwpm ratio b c n_disc = $(eval_row "$SD"); best epoch $(best_epoch "$SD"); $(grep -m1 -o 'params=[0-9,]*' "$SD/run.log"); clipped $(grep -o 'cumulative [0-9.]*' "$SD/run.log" | tail -1)"
}
sample_resources () {
  local t names c
  t=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
  echo -e "$t\thost_available_gb\t$(host_avail_gb)" >> "$RES"
  podman stats --no-stream --format '{{.Name}}\t{{.MemUsage}}' 2>/dev/null | grep '^sc_' \
    | sed "s/^/$t\tcontainer_mem\t/" >> "$RES"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | sed "s/^/$t\tgpu_pid_mib\t/" >> "$RES"
}

log "BEGIN scaling100M driver pid=$$ repo=$(git -C "$REPO" rev-parse --short HEAD) global_clipnorm=$GCLIP free GPU $(free_mib) MiB host available $(host_avail_gb) GB"
until [ "$(free_mib)" -ge "$NEED_FREE" ]; do sleep 60; done
launch L1_d7_u280_100M_s0 7 280 0 \
  "${P7}_TRAINONLY_seed43.npz ${P7}_TRAINONLY_seed44.npz ${P7}_TRAINONLY_seed45.npz" \
  "5000000 30000000 50000000"
S1=$(seeddir L1_d7_u280_100M_s0 0)

# L2 waits until L1 has finished loading its data and is training (first epoch line or
# the "Epoch 1/80" header), so the free-memory checks see L1 at its full size.
L2_STARTED=""
until [ -n "$L2_STARTED" ]; do
  sample_resources
  if ! running sc_L1_d7_u280_100M_s0; then L2_STARTED=no; log "NOT STARTING L2: L1 is not running"; break; fi
  if grep -q 'Epoch 1/80' "$S1/run.log" 2>/dev/null; then
    sleep 120; sample_resources
    if [ "$(free_mib)" -ge "$NEED_FREE" ] && [ "$(host_avail_gb)" -ge "$NEED_HOST_GB" ]; then
      launch L2_d5_u140_20M_s0 5 140 0 "${P5}_TRAINONLY_seed43.npz" "5000000"; L2_STARTED=yes
    else
      L2_STARTED=no
      log "NOT STARTING L2: free GPU $(free_mib) MiB (need $NEED_FREE), host available $(host_avail_gb) GB (need $NEED_HOST_GB)"
    fi
  fi
  sleep 30
done

L1=""; L2=""; [ "$L2_STARTED" = yes ] || L2=skip
while :; do
  sample_resources
  running sc_L1_d7_u280_100M_s0 && clip_check L1_d7_u280_100M_s0 0
  [ -z "$L2" ] && running sc_L2_d5_u140_20M_s0 && clip_check L2_d5_u140_20M_s0 0
  if [ -z "$L1" ] && ! running sc_L1_d7_u280_100M_s0; then L1=yes; finish L1_d7_u280_100M_s0 0; fi
  if [ -z "$L2" ] && ! running sc_L2_d5_u140_20M_s0; then L2=yes; finish L2_d5_u140_20M_s0 0; fi
  [ -n "$L1" ] && [ -n "$L2" ] && break
  sleep 30
done

# Pairings (only for completed runs)
if [ -f "$S1/COMPLETE" ]; then
  pair "$S1" "$(ls "$K1_DIR"/per_shot_*.npz)" L1_d7_u280_100M_s0 K1_cos_u280_50M_s0
  pair "$S1" "$(ls "$K2_DIR"/per_shot_*.npz)" L1_d7_u280_100M_s0 K2_cos_u280_50M_s1
fi
if [ "$L2_STARTED" = yes ]; then
  S2=$(seeddir L2_d5_u140_20M_s0 0)
  [ -f "$S2/COMPLETE" ] && pair "$S2" "$(ls "$CTL_DIR"/per_shot_*.npz)" L2_d5_u140_20M_s0 control_const_d5_u140_20M_s0
fi
log "END   scaling100M queue finished"
