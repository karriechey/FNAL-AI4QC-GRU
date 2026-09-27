#!/usr/bin/env bash
# Created: 2026-09-27
# Last updated: 2026-09-27
#
# Morning queue 2026-09-27, run unattended on grace1 alongside run C of the overnight plan.
#
#   1  d7 u280 reset_after=False 20M seed 1 (FORMAL 15M + seed43 5M): a second seed for the
#      width result (seed 0 = overnight run B)
#   2  d7 u280 reset_after=False 50M seed 0 (FORMAL 15M + seed43 5M + seed44 30M): width and
#      data together
#
# Every GPU job: batch 10,000, 200 epochs, LR 3e-3 constant, early stopping off, determinism
# on, TF32 on for training / off for evaluation, best checkpoint by validation LER, 1.6 GB
# GPU cap; a job starts only with >= 2 GB GPU memory free; at most 3 training jobs.
# Best epoch >= 180 is flagged "possibly under-trained".
#
#     setsid nohup bash driver.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail

REPO="$HOME/FNAL-AI4QC-GRU"
DIR="$REPO/experiments/morning_2026-09-27"
OVN="$REPO/experiments/overnight_2026-09-27"
LOG="$DIR/driver.log"
IMG=qdec:tf215
PYH="$HOME/tfenv/bin/python"
CAP=1600
NEED_FREE=2000
OLD_D7S0="$REPO/experiments/queue_ra1_20M/old_rescored_tf32off/d7_s0/per_shot_old_ra0.npz"
OLD_D7S1="$REPO/experiments/queue_ra1_20M/old_rescored_tf32off/d7_s1/per_shot_old_ra0.npz"
B_U280_S0=$(ls "$OVN"/B_d7_u280_20M_*/seed0/per_shot_*.npz)
P7="$HOME/pools_d7_p004_19M/data_d7_p0.004_r7"
mkdir -p "$DIR"

log () { echo "[morning $(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG"; }
free_mib () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
n_training () { podman ps --format '{{.Command}}' | grep -c 'ROUNDS='; }
running () { [ -n "$(podman ps -q --filter name="$1")" ]; }

launch () {   # launch <name> <d> <units> <extra_pools> <extra_ns> <seed>
  local name="$1" D="$2" U="$3" XP="$4" XN="$5" SEED="$6" S OUT
  S=$(date -u +%Y%m%dT%H%M%SZ); OUT="$DIR/${name}_$S"; mkdir -p "$OUT"
  { echo "at_start repo_head $(git -C "$REPO" rev-parse HEAD)"
    echo "at_start train_student_sha256 $(sha256sum "$REPO/src/train_student.py" | cut -d' ' -f1)"
  } > "$OUT/code_commit.txt"
  echo "$OUT" > "$DIR/$name.out"
  log "START $name d=$D units=$U seed=$SEED extra=[$XP] n=[$XN] -> $OUT (free GPU $(free_mib) MiB)"
  (podman run --rm --name "ovn_$name" --security-opt=label=disable --device nvidia.com/gpu=all \
     -v "$HOME:$HOME" -e HOME="$HOME" -w "$REPO/src" "$IMG" \
     env D="$D" ROUNDS="$D" SEEDS="$SEED" UNITS="$U" NTRAIN=15000000 EXTRA_POOL="$XP" EXTRA_N="$XN" \
         BATCH=10000 EPOCHS=200 LR=0.003 EARLY_STOP=0 RESET_AFTER=0 UNROLL=0 EVAL_NO_TF32=1 \
         GPU_MEM_MIB="$CAP" MIN_FREE_MIB="$CAP" OUT="$OUT" STAMP="$S" \
     bash gh200_run_student_scaling.sh > "$OUT/podman.log" 2>&1; echo $? > "$OUT/exit_code") &
}

eval_row () {  # eval_row <seeddir> -> "p_L mwpm ratio b c nd"
  python3 -c "
import csv,sys; r=list(csv.DictReader(open('$1/eval_best_ckpt.csv')))[-1]
print(r['p_L'], r['mwpm_p_L'], r['ratio'], r['student_only'], r['mwpm_only'], r['n_discordant'])"
}
best_epoch () { python3 -c "import json,glob;print(json.load(open(glob.glob('$1/ckpt/*.best.json')[0]))['best_epoch'])"; }
pair () {      # pair <new dump> <other dump> <a name> <b name> <csv>
  ( cd "$REPO/src" && "$PYH" pair_two_decoders.py --a "$1" --b "$2" --a-name "$3" --b-name "$4" \
      --a-column student_correct --b-column student_correct --out-csv "$5" ) >> "$DIR/pairing.log" 2>&1
}
postprocess () {  # postprocess <name> <seed> <pairs...> as "dump|bname"
  local name="$1" seed="$2"; shift 2
  local OUT SD BE
  OUT=$(cat "$DIR/$name.out"); SD="$OUT/seed$seed"
  if [ ! -f "$SD/COMPLETE" ]; then log "FAIL  $name ended without COMPLETE (exit $(cat "$OUT/exit_code" 2>/dev/null))"; return; fi
  BE=$(best_epoch "$SD")
  log "DONE  $name: eval p_L mwpm ratio b c n_disc = $(eval_row "$SD"); best epoch $BE; $(grep -m1 -o 'params=[0-9,]*' "$SD/run.log")"
  [ "$BE" -ge 180 ] && { log "FLAG  $name best epoch $BE >= 180: possibly under-trained"; echo "best epoch $BE" > "$OUT/FLAG_possibly_under_trained"; }
  for pr in "$@"; do
    local dump="${pr%%|*}" bname="${pr##*|}"
    pair "$(ls "$SD"/per_shot_*.npz)" "$dump" "$name" "$bname" "$SD/mcnemar_vs_$bname.csv" \
      && log "PAIR  $name vs $bname -> $SD/mcnemar_vs_$bname.csv" || log "PAIR  $name vs $bname FAILED"
  done
}

log "BEGIN morning driver pid=$$ repo=$(git -C "$REPO" rev-parse --short HEAD)"
JOBS=("M1_d7_u280_20M_s1|7|280|${P7}_TRAINONLY_seed43.npz|5000000|1"
      "M2_d7_u280_50M_s0|7|280|${P7}_TRAINONLY_seed43.npz ${P7}_TRAINONLY_seed44.npz|5000000 30000000|0")
for j in "${JOBS[@]}"; do
  IFS='|' read -r name D U XP XN SEED <<< "$j"
  until [ "$(free_mib)" -ge "$NEED_FREE" ] && [ "$(n_training)" -lt 3 ]; do sleep 60; done
  launch "$name" "$D" "$U" "$XP" "$XN" "$SEED"
  sleep 300          # let the job allocate before measuring free memory again
done
D1=""; D2=""
while [ -z "$D1" ] || [ -z "$D2" ]; do
  if [ -z "$D1" ] && ! running ovn_M1_d7_u280_20M_s1; then
    D1=yes; postprocess M1_d7_u280_20M_s1 1 "$B_U280_S0|u280_20M_s0" "$OLD_D7S1|old_ra0_d7_u140_s1"
  fi
  if [ -z "$D2" ] && ! running ovn_M2_d7_u280_50M_s0; then
    D2=yes; postprocess M2_d7_u280_50M_s0 0 "$B_U280_S0|u280_20M_s0" "$OLD_D7S0|old_ra0_d7_u140_s0"
  fi
  sleep 60
done
log "END   morning jobs finished (pair M2 with overnight run C when C is complete)"
