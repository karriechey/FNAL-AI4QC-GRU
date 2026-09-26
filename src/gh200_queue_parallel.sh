#!/usr/bin/env bash
# Created: 2026-09-26
# Last updated: 2026-09-26
#
# Parallel GPU scheduler for the reset_after=True queue (Karrie, 2026-09-26). Replaces the
# sequential gh200_queue_ra1_20M.sh, whose running d5 s0 job it adopts without restarting.
#
#   jobs         read from $QROOT/jobs.tsv every minute, in file order, so jobs can be
#                appended or made ready while the scheduler runs. Columns (tab-separated):
#                  name  d  seed  extra_seeds  extra_ns  ready_marker  pair_with
#                extra_seeds / extra_ns: comma lists of TRAINONLY seeds and shot counts
#                (e.g. 43,44 and 5000000,30000000); ready_marker: '-' or a file that must
#                exist before the job may start (the 50M job waits for its code change);
#                pair_with: '-' or a job name whose new checkpoint it is also paired with.
#                A job that is not ready is passed over, and the next ready one starts.
#   parallelism  up to MAX_PAR=3 jobs, each capped at 1,600 MiB of GPU memory. A job starts
#                only if nvidia-smi reports at least 2,000 MiB free; otherwise wait. If a job
#                runs out of GPU memory, MAX_PAR drops to 2 and the job is queued again once.
#   every job    u140, reset_after=True, unroll=True, FORMAL [0,15M) + its TRAINONLY pools,
#                batch 10,000, 200 epochs, LR 3e-3, determinism on, TF32 on for training /
#                off for evaluation, best checkpoint by val LER (launcher settings).
#   after a job  (1) code_commit.txt: repo HEAD and the last commit touching train_student.py
#                (2) best epoch >= 180 is flagged "possibly under-trained"
#                (3) the old reset_after=False checkpoint of the same d and seed is re-scored
#                    with TF32 off on the eval block and paired with the new one (McNemar)
#                (4) paired with the pair_with job's checkpoint, if given
#                (5) d5 s0 only: C2b, hls4ml vs Keras for the trained reset_after=True,
#                    unroll=True checkpoint (image, CPU, 65,536-entry ap_fixed<32,16> tables)
#   The sealed partition is never read.
#
#     setsid nohup bash gh200_queue_parallel.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail

REPO="$HOME/FNAL-AI4QC-GRU"
QROOT="$REPO/experiments/queue_ra1_20M"
JOBS="$QROOT/jobs.tsv"
QLOG="$QROOT/queue.log"
STATUS="$QROOT/status.tsv"
IMG="${IMG:-qdec:tf215}"
PYH="$HOME/tfenv/bin/python"
MAX_PAR=3
CAP_MIB=1600
NEED_FREE_MIB=2000
STATE="$QROOT/state"          # one file per job: started / done / failed / retried
mkdir -p "$STATE"
declare -A OLDRUN=( [5]=d5_u140_ntr20M_aug_20260821T154820Z [7]=d7_u140_ntr20M_aug_20260821T193446Z )

log () { echo "[sched $(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$QLOG"; }
free_mib () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

postprocess () {   # postprocess <name> <d> <seed> <out> <pair_with>
  local name="$1" D="$2" SEED="$3" OUT="$4" PAIRW="$5"
  local SEEDDIR="$OUT/seed$SEED"
  { echo "repo_head $(git -C "$REPO" rev-parse HEAD)"
    echo "train_student_last_commit $(git -C "$REPO" log -1 --format=%H -- src/train_student.py)"
  } > "$OUT/code_commit.txt"
  local BJ BEST_EP
  BJ=$(ls "$SEEDDIR"/ckpt/*.best.json 2>/dev/null | head -1)
  BEST_EP=$([ -n "$BJ" ] && python3 -c "import json;print(json.load(open('$BJ'))['best_epoch'])" || echo "?")
  if [ "$BEST_EP" != "?" ] && [ "$BEST_EP" -ge 180 ]; then
    log "FLAG  $name best epoch $BEST_EP >= 180: possibly under-trained"
    echo "possibly under-trained: best epoch $BEST_EP of 200" > "$OUT/FLAG_possibly_under_trained"
  else
    log "BEST  $name best epoch $BEST_EP"
  fi
  local NEW RES OLDCK OLD=""
  NEW=$(ls "$SEEDDIR"/per_shot_*.npz | head -1)
  OLDCK="$HOME/rcnn_threshold/results/${OLDRUN[$D]}/seed$SEED/ckpt/gru_d${D}_p004_u140_ext5000000_hard_seed${SEED}_ntr15000000_b10000.best.weights.h5"
  RES="$QROOT/old_rescored_tf32off/d${D}_s${SEED}"
  if [ -f "$RES/per_shot_old_ra0.npz" ]; then
    OLD="$RES/per_shot_old_ra0.npz"
  elif [ -f "$OLDCK" ]; then
    mkdir -p "$RES"
    until [ "$(free_mib)" -ge "$NEED_FREE_MIB" ]; do sleep 60; done
    eval "$("$PYH" "$REPO/src/partitions.py" --shell --d "$D")"
    podman run --rm --security-opt=label=disable --device nvidia.com/gpu=all \
      -v "$HOME:$HOME" -e HOME="$HOME" -w "$REPO/src" "$IMG" \
      python3 eval_student_on_tail.py --student gru --inputs evts --hidden --units 140 \
        --no-tf32 --gpu-mem-mib "$CAP_MIB" --weights "$OLDCK" \
        --d "$D" --p 0.004 --rounds "$D" --pool "$POOL" \
        --eval-start "$EVAL_START" --n-test "$EVAL_N" --batch-size 10000 \
        --dump-per-shot "$RES/per_shot_old_ra0.npz" --out-csv "$RES/eval_old_ra0.csv" \
        > "$RES/rescore.log" 2>&1 \
      && OLD="$RES/per_shot_old_ra0.npz" && log "RESCORE old d=$D seed=$SEED (TF32 off) -> $RES" \
      || log "RESCORE old d=$D seed=$SEED FAILED (see $RES/rescore.log)"
  fi
  if [ -n "$OLD" ]; then
    ( cd "$REPO/src" && "$PYH" pair_two_decoders.py --a "$NEW" --b "$OLD" \
        --a-name "$name" --b-name "ra0_d${D}_s${SEED}" \
        --a-column student_correct --b-column student_correct \
        --out-csv "$SEEDDIR/mcnemar_vs_reset_after_false.csv" ) >> "$SEEDDIR/pair.log" 2>&1 \
      && log "PAIR  $name vs old ra0 d${D} s${SEED}" || log "PAIR  $name vs old FAILED"
  fi
  if [ "$PAIRW" != "-" ]; then
    local PDIR PNEW
    PDIR=$(cat "$STATE/$PAIRW.out" 2>/dev/null)
    PNEW=$(ls "$PDIR"/seed*/per_shot_*.npz 2>/dev/null | head -1)
    if [ -n "$PNEW" ]; then
      ( cd "$REPO/src" && "$PYH" pair_two_decoders.py --a "$NEW" --b "$PNEW" \
          --a-name "$name" --b-name "$PAIRW" \
          --a-column student_correct --b-column student_correct \
          --out-csv "$SEEDDIR/mcnemar_vs_$PAIRW.csv" ) >> "$SEEDDIR/pair.log" 2>&1 \
        && log "PAIR  $name vs $PAIRW" || log "PAIR  $name vs $PAIRW FAILED"
    else
      log "PAIR  $name vs $PAIRW: $PAIRW has no per-shot dump yet; pair later"
    fi
  fi
  if [ "$name" = "d5_s0" ]; then
    local C2B BEST
    C2B="$REPO/experiments/gru_check/c2b_$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$C2B"
    BEST=$(ls "$SEEDDIR"/ckpt/*.best.weights.h5 | head -1)
    podman run --rm --security-opt=label=disable -v "$HOME:$HOME" -e HOME="$HOME" \
      -e TF_CPP_MIN_LOG_LEVEL=3 -w "$REPO" "$IMG" \
      python3 tests/c2_hls4ml_gru_check.py --out "$C2B" --trained-weights "$BEST" \
        --trained-units 140 --trained-reset-after --trained-unroll --no-tf32 --table-size 65536 \
        --table-precision "ap_fixed<32,16>" > "$C2B/run.log" 2>&1 \
      && log "C2B   done on $BEST -> $C2B" || log "C2B   FAILED (see $C2B/run.log)"
  fi
}

run_job () {       # run_job <name> <d> <seed> <extra_seeds> <extra_ns> <pair_with>
  local name="$1" D="$2" SEED="$3" XS="$4" XN="$5" PAIRW="$6"
  local STAMP OUT EXTRA_POOL EXTRA_N s
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  OUT="$QROOT/${name}_${STAMP}"
  mkdir -p "$OUT"; echo "$OUT" > "$STATE/$name.out"
  EXTRA_POOL=""; for s in ${XS//,/ }; do
    EXTRA_POOL="$EXTRA_POOL $HOME/pools_d${D}_p004_19M/data_d${D}_p0.004_r${D}_TRAINONLY_seed${s}.npz"; done
  EXTRA_POOL="${EXTRA_POOL# }"; EXTRA_N="${XN//,/ }"
  log "RUN   $name d=$D seed=$SEED extra=[$XS] n=[$XN] cap=${CAP_MIB}MiB -> $OUT"
  local t0 rc dt; t0=$(date +%s)
  podman run --rm --security-opt=label=disable --device nvidia.com/gpu=all \
    -v "$HOME:$HOME" -e HOME="$HOME" -w "$REPO/src" "$IMG" \
    env D="$D" ROUNDS="$D" SEEDS="$SEED" UNITS=140 NTRAIN=15000000 \
        EXTRA_POOL="$EXTRA_POOL" EXTRA_N="$EXTRA_N" BATCH=10000 EPOCHS=200 LR=0.003 \
        EARLY_STOP=0 RESET_AFTER=1 UNROLL=1 EVAL_NO_TF32=1 \
        GPU_MEM_MIB="$CAP_MIB" MIN_FREE_MIB="$CAP_MIB" OUT="$OUT" STAMP="$STAMP" \
    bash gh200_run_student_scaling.sh > "$OUT/podman.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ] && [ -f "$OUT/seed$SEED/COMPLETE" ]; then
    log "DONE  $name in ${dt}s"
    printf '%s\t%s\t%s\tdone\t%s\t%s\n' "$name" "$D" "$SEED" "$dt" "$OUT" >> "$STATUS"
    touch "$STATE/$name.done"
    postprocess "$name" "$D" "$SEED" "$OUT" "$PAIRW"
  elif grep -qE "ResourceExhausted|OOM when allocating|out of memory|Dst tensor is not initialized" \
        "$OUT/podman.log" "$OUT/seed$SEED/run.log" 2>/dev/null; then
    log "OOM   $name after ${dt}s"
    printf '%s\t%s\t%s\tfailed_oom\t%s\t%s\n' "$name" "$D" "$SEED" "$dt" "$OUT" >> "$STATUS"
    touch "$STATE/$name.oom"
  else
    log "FAIL  $name rc=$rc after ${dt}s"
    printf '%s\t%s\t%s\tfailed\t%s\t%s\n' "$name" "$D" "$SEED" "$dt" "$OUT" >> "$STATUS"
    touch "$STATE/$name.failed"
  fi
}

adopt_running () {  # adopt_running <name> <d> <seed> <out>  (job started by the old driver)
  local name="$1" D="$2" SEED="$3" OUT="$4"
  echo "$OUT" > "$STATE/$name.out"
  log "ADOPT $name (already running) -> $OUT"
  until [ -f "$OUT/seed$SEED/COMPLETE" ] || ! podman ps --format '{{.Command}}' | grep -q "D=$D ROUNDS=$D SEEDS=$SEED"; do
    sleep 60; done
  if [ -f "$OUT/seed$SEED/COMPLETE" ]; then
    log "DONE  $name (adopted)"
    printf '%s\t%s\t%s\tdone\t-\t%s\n' "$name" "$D" "$SEED" "$OUT" >> "$STATUS"
    touch "$STATE/$name.done"
    postprocess "$name" "$D" "$SEED" "$OUT" "-"
  else
    log "FAIL  $name (adopted): container ended without COMPLETE"
    touch "$STATE/$name.failed"
  fi
}

log "START scheduler pid=$$ max_par=$MAX_PAR cap=${CAP_MIB}MiB need_free=${NEED_FREE_MIB}MiB repo=$(git -C "$REPO" rev-parse --short HEAD)"
if [ -n "${ADOPT:-}" ]; then     # ADOPT="name d seed out"
  read -r an ad as ao <<< "$ADOPT"
  touch "$STATE/$an.started"
  adopt_running "$an" "$ad" "$as" "$ao" &
fi
declare -A PIDS=()
while :; do
  for n in "${!PIDS[@]}"; do kill -0 "${PIDS[$n]}" 2>/dev/null || unset "PIDS[$n]"; done
  # OOM fallback: drop to 2 in parallel and queue the job once more.
  for f in "$STATE"/*.oom; do
    [ -e "$f" ] || continue
    n=$(basename "$f" .oom)
    if [ ! -e "$STATE/$n.retried" ]; then
      MAX_PAR=2; log "OOM   fallback: max_par=2; $n queued again"
      rm -f "$STATE/$n.started" "$f"; touch "$STATE/$n.retried"
    fi
  done
  # training containers (launcher jobs, including the adopted one); rescore/C2b are not counted
  running=$(podman ps --format '{{.Command}}' | grep -c "ROUNDS=")
  pending=0; next=""
  while IFS=$'\t' read -r name d seed xs xn ready pairw; do
    [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
    [ -e "$STATE/$name.started" ] && continue
    pending=$((pending + 1))
    if [ -z "$next" ] && { [ "$ready" = "-" ] || [ -e "$ready" ]; }; then
      next="$name	$d	$seed	$xs	$xn	$pairw"
    fi
  done < "$JOBS"
  if [ $pending -eq 0 ] && [ $running -eq 0 ] && [ ${#PIDS[@]} -eq 0 ]; then
    log "END   no jobs left"; break
  fi
  if [ -n "$next" ] && [ "$running" -lt "$MAX_PAR" ]; then
    fm=$(free_mib)
    if [ "$fm" -ge "$NEED_FREE_MIB" ]; then
      IFS=$'\t' read -r name d seed xs xn pairw <<< "$next"
      touch "$STATE/$name.started"
      run_job "$name" "$d" "$seed" "$xs" "$xn" "$pairw" &
      PIDS[$name]=$!
      sleep 180          # let the new job allocate before measuring free memory again
      continue
    fi
  fi
  sleep 60
done
wait
