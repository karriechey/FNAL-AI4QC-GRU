#!/usr/bin/env bash
# Created: 2026-09-27
# Last updated: 2026-09-27
#
# Overnight plan 2026-09-27 (pre-registered before any result), run unattended on grace1.
#
#   control   d5 seed 0, new pipeline, reset_after=False (already running, started 06:35:55Z)
#             PASS if eval p_L <= 0.00730 (old reset_after=False seeds 0.007119-0.007247 plus
#             margin). McNemar b / c / n_discordant are reported but are not the pass rule.
#   A (CPU)   d7 50M data sanity check (sanity_d7_50M.py); pass = labels and inputs from the
#             multi-pool loader equal direct reads, and per-part flip rate / density within
#             1 % and MWPM error rate within 3 standard errors of the FORMAL part.
#   B (GPU)   d7 u280 reset_after=False, 20M augmented, seed 0 - starts now, alongside the
#             control. If the control FAILS, B is stopped and the reason logged.
#   C (GPU)   d7 u140 reset_after=False, 50M (FORMAL 15M + seed43 5M + seed44 30M), seed 0 -
#             starts only if the control PASSES and A PASSES; otherwise not started.
#
# Every GPU job: batch 10,000, 200 epochs, LR 3e-3 constant, early stopping off, determinism
# on, TF32 on for training / off for evaluation, best checkpoint by validation LER, 1.6 GB
# GPU cap; a job starts only with >= 2 GB GPU memory free; at most 3 training jobs.
# Best epoch >= 180 is flagged "possibly under-trained".
#
#     setsid nohup bash driver.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail

REPO="$HOME/FNAL-AI4QC-GRU"
DIR="$REPO/experiments/overnight_2026-09-27"
LOG="$DIR/driver.log"
IMG=qdec:tf215
PYH="$HOME/tfenv/bin/python"
CAP=1600
NEED_FREE=2000
CTRL="$REPO/experiments/control_ra0/d5_s0_20260927T063555Z"
PASS_RULE=0.00730
OLD_D5S0="$REPO/experiments/queue_ra1_20M/old_rescored_tf32off/d5_s0/per_shot_old_ra0.npz"
RA1_D5S0=$(ls "$REPO"/experiments/queue_ra1_20M/d5_s0_20260926T174023Z/seed0/per_shot_*.npz)
OLD_D7S0="$REPO/experiments/queue_ra1_20M/old_rescored_tf32off/d7_s0/per_shot_old_ra0.npz"
RA1_D7S0_50M=$(ls "$REPO"/experiments/queue_ra1_20M/d7_s0_50M_*/seed0/per_shot_*.npz)
P7="$HOME/pools_d7_p004_19M/data_d7_p0.004_r7"
mkdir -p "$DIR"

log () { echo "[overnight $(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG"; }
free_mib () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
n_training () { podman ps --format '{{.Command}}' | grep -c 'ROUNDS='; }
running () { [ -n "$(podman ps -q --filter name="$1")" ]; }

launch () {   # launch <name> <d> <units> <extra_pools> <extra_ns>
  local name="$1" D="$2" U="$3" XP="$4" XN="$5" S OUT
  S=$(date -u +%Y%m%dT%H%M%SZ); OUT="$DIR/${name}_$S"; mkdir -p "$OUT"
  { echo "at_start repo_head $(git -C "$REPO" rev-parse HEAD)"
    echo "at_start train_student_sha256 $(sha256sum "$REPO/src/train_student.py" | cut -d' ' -f1)"
  } > "$OUT/code_commit.txt"
  echo "$OUT" > "$DIR/$name.out"
  log "START $name d=$D units=$U extra=[$XP] n=[$XN] -> $OUT (free GPU $(free_mib) MiB)"
  (podman run --rm --name "ovn_$name" --security-opt=label=disable --device nvidia.com/gpu=all \
     -v "$HOME:$HOME" -e HOME="$HOME" -w "$REPO/src" "$IMG" \
     env D="$D" ROUNDS="$D" SEEDS=0 UNITS="$U" NTRAIN=15000000 EXTRA_POOL="$XP" EXTRA_N="$XN" \
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
postprocess () {  # postprocess <name> <pairs...> as "dump|bname"
  local name="$1"; shift
  local OUT SD BE
  OUT=$(cat "$DIR/$name.out"); SD="$OUT/seed0"
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

log "BEGIN driver pid=$$ repo=$(git -C "$REPO" rev-parse --short HEAD)"

# A: sanity check on the CPU
( podman run --rm --name ovn_A_sanity --security-opt=label=disable -v "$HOME:$HOME" -e HOME="$HOME" \
    -e TF_CPP_MIN_LOG_LEVEL=3 -e CUDA_VISIBLE_DEVICES= -w "$DIR" "$IMG" \
    python3 sanity_d7_50M.py --out "$DIR/A_sanity" > "$DIR/A_sanity.log" 2>&1 ) &
log "START A (CPU) sanity_d7_50M.py -> $DIR/A_sanity"

# B: now, alongside the control
until [ "$(free_mib)" -ge "$NEED_FREE" ] && [ "$(n_training)" -lt 3 ]; do sleep 60; done
launch B_d7_u280_20M 7 280 "${P7}_TRAINONLY_seed43.npz" 5000000

CTRL_V=""; A_V=""; C_STARTED=""; B_DONE=""; C_DONE=""
while :; do
  # control verdict
  if [ -z "$CTRL_V" ] && [ -f "$CTRL/seed0/COMPLETE" ]; then
    read -r pL mw ratio b c nd <<< "$(eval_row "$CTRL/seed0")"
    CTRL_V=$(python3 -c "print('PASS' if float('$pL') <= $PASS_RULE else 'FAIL')")
    log "CONTROL eval p_L $pL (MWPM $mw, ${ratio}x; b $b c $c n_disc $nd), best epoch $(best_epoch "$CTRL/seed0"): $CTRL_V (rule p_L <= $PASS_RULE)"
    pair "$(ls "$CTRL"/seed0/per_shot_*.npz)" "$OLD_D5S0" control_ra0_s0 old_ra0_s0 "$CTRL/seed0/mcnemar_vs_old_ra0_s0.csv" && log "PAIR  control vs old reset_after=False seed 0"
    pair "$(ls "$CTRL"/seed0/per_shot_*.npz)" "$RA1_D5S0" control_ra0_s0 ra1_s0 "$CTRL/seed0/mcnemar_vs_ra1_s0.csv" && log "PAIR  control vs reset_after=True seed 0"
    echo "$CTRL_V" > "$DIR/CONTROL_$CTRL_V"
    if [ "$CTRL_V" = FAIL ] && running ovn_B_d7_u280_20M; then
      podman stop -t 30 ovn_B_d7_u280_20M >/dev/null
      log "STOP  B: control FAILED the pre-registered rule (p_L $pL > $PASS_RULE)"
    fi
  fi
  # A verdict
  if [ -z "$A_V" ] && [ -f "$DIR/A_sanity/sanity_result.json" ]; then
    A_V=$(python3 -c "import json;print('PASS' if json.load(open('$DIR/A_sanity/sanity_result.json'))['passed'] else 'FAIL')")
    log "A     sanity check: $A_V (details $DIR/A_sanity/sanity_result.json)"
  elif [ -z "$A_V" ] && ! running ovn_A_sanity && [ ! -f "$DIR/A_sanity/sanity_result.json" ]; then
    A_V=FAIL; log "A     sanity check container ended without a result: FAIL (see A_sanity.log)"
  fi
  # C
  if [ -z "$C_STARTED" ] && [ -n "$CTRL_V" ] && [ -n "$A_V" ]; then
    if [ "$CTRL_V" = PASS ] && [ "$A_V" = PASS ]; then
      if [ "$(free_mib)" -ge "$NEED_FREE" ] && [ "$(n_training)" -lt 3 ]; then
        launch C_d7_u140_50M 7 140 "${P7}_TRAINONLY_seed43.npz ${P7}_TRAINONLY_seed44.npz" "5000000 30000000"
        C_STARTED=yes
      fi
    else
      log "NOT STARTING C: control $CTRL_V, sanity $A_V"; C_STARTED=no
    fi
  fi
  # finished jobs
  if [ -z "$B_DONE" ] && [ -f "$DIR/B_d7_u280_20M.out" ] && ! running ovn_B_d7_u280_20M; then
    B_DONE=yes; postprocess B_d7_u280_20M "$OLD_D7S0|old_ra0_d7_u140_s0"
  fi
  if [ "$C_STARTED" = yes ] && [ -z "$C_DONE" ] && ! running ovn_C_d7_u140_50M; then
    C_DONE=yes; postprocess C_d7_u140_50M "$OLD_D7S0|old_ra0_d7_u140_s0" "$RA1_D7S0_50M|ra1_d7_s0_50M"
  fi
  if [ -n "$CTRL_V" ] && [ -n "$A_V" ] && [ -n "$B_DONE" ] && { [ "$C_STARTED" = no ] || [ -n "$C_DONE" ]; }; then
    log "END   all overnight jobs finished"; break
  fi
  sleep 60
done
