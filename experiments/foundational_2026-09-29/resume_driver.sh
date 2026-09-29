#!/usr/bin/env bash
# Created: 2026-09-29
# Last updated: 2026-09-29
#
# Resume F2-F5 of this queue from epoch 100 to 200 (requested 2026-09-29 after all four
# had their best checkpoint at epoch 98-100 with validation loss still falling).
#
# Each resumed run gets its own new folder <name>_resume200_<stamp>; the original folders
# are only read. Before launch the driver copies the original best checkpoint
# (.best.keras/.best.weights.h5/.best.json) into the new ckpt/ and the original
# .lastepoch.weights.h5 into resume_from/; training then continues with
#   RESUME_WEIGHTS  (model + Adam state after epoch 100)   INITIAL_EPOCH=100   EPOCHS=200
#   RESUME_BEST_JSON (the best so far; replaced only by a better epoch)
#   RESUME_HISTORY   (epochs 1-100; the written history covers 1-200)
# and every other setting as in the original run (same data, seed, LR, batch, reset_after,
# val-LER checkpoint, evaluation block with MWPM on the same shots). The per-epoch shuffle
# is advanced past epoch 100, so epochs 101-200 see the data order an uninterrupted run
# would (tests/resume_exactness.sh). The sealed partition [17M, 19M) is not read.
#
# Slots, thresholds and logging as in driver2.sh (MAX_JOBS default 4; start needs host
# >= 150 GB and free GPU >= 2,500 MiB with another job running). New: each sample also
# records every container's host process ids (container_hpids), so summary.json takes the
# GPU peak from the run's own processes.
#
#     setsid nohup bash resume_driver.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail

REPO="$HOME/FNAL-AI4QC-GRU"
DIR="$REPO/experiments/foundational_2026-09-29"
LOG="$DIR/driver.log"
RES="$DIR/resources.tsv"
IMG=qdec:tf215
PYH="$HOME/tfenv/bin/python"
CAP=1600
NEED_FREE_ALONE=2000
NEED_FREE_SECOND=2500
NEED_HOST_GB=150
MAX_JOBS="${MAX_JOBS:-4}"
GCLIP=0.64
CLIP_LIMIT=0.05
P7="$HOME/pools_d7_p004_19M/data_d7_p0.004_r7"

# Earlier locks (driver.lock, driver2.lock) stay held by podman processes that inherited
# the descriptor, so this driver uses resume_driver.lock and closes it for every podman it starts.
exec 9> "$DIR/resume_driver.lock"
flock -n 9 || { echo "[foundational $(date -u '+%Y-%m-%dT%H:%M:%SZ')] another resume_driver holds resume_driver.lock; exiting" >> "$LOG"; exit 1; }

log () { echo "[foundational $(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG"; }
free_mib () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
host_avail_gb () { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
running () { [ -n "$(podman ps -q --filter name="$1")" ]; }
sample_resources () {
  local t; t=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
  echo -e "$t\thost_available_gb\t$(host_avail_gb)" >> "$RES"
  podman stats --no-stream --format '{{.Name}}\t{{.MemUsage}}' 2>/dev/null | grep '^fd_' \
    | sed "s/^/$t\tcontainer_mem\t/" >> "$RES"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | sed "s/^/$t\tgpu_pid_mib\t/" >> "$RES"
  local c
  for c in $(podman ps --format '{{.Names}}' | grep '^fd_'); do
    echo -e "$t\tcontainer_hpids\t$c\t$(podman top "$c" hpid 2>/dev/null | tail -n +2 | tr '\n' ' ')" >> "$RES"
  done
}

# Queue, in order: name source-run units lr schedule
QUEUE=("F2_d7_u70_20M_resume200 F2_d7_u70_20M 70 0.003 constant"
       "F3_d7_u200_20M_resume200 F3_d7_u200_20M 200 0.003 constant"
       "F4_d7_u140_20M_lr3e-4_resume200 F4_d7_u140_20M_lr3e-4 140 0.0003 constant"
       "F5_d7_u140_20M_lr3e-5_resume200 F5_d7_u140_20M_lr3e-5 140 0.00003 constant")
declare -A PID OUTD SCHED      # per active job: podman pid ("adopted" for F1), folder, schedule

launch () {   # launch <name> <source run> <units> <lr> <schedule>
  local name="$1" SRCN="$2" U="$3" LRV="$4" SCH="$5" S OUT SRC TAGB i ENVX
  SRC=$(ls -d "$DIR/${SRCN}"_2* | head -1)
  TAGB=$(basename "$(ls "$SRC"/seed0/ckpt/*.best.json)" .best.json)
  S=$(date -u +%Y%m%dT%H%M%SZ); OUT="$DIR/${name}_$S"; mkdir -p "$OUT/seed0/ckpt" "$OUT/resume_from"
  cp -p "$SRC/seed0/ckpt/$TAGB".best.{keras,weights.h5,json} "$OUT/seed0/ckpt/"
  cp -p "$SRC/seed0/ckpt/$TAGB.lastepoch.weights.h5" "$OUT/resume_from/"
  cp -p "$SRC/seed0/runs/$TAGB.history.json" "$OUT/resume_from/"
  { echo "at_start repo_head $(git -C "$REPO" rev-parse HEAD)"
    echo "at_start train_student_sha256 $(sha256sum "$REPO/src/train_student.py" | cut -d' ' -f1)"
    echo "resumed_from $SRC (epoch 100)"
    (cd "$OUT" && sha256sum resume_from/* seed0/ckpt/*)
  } > "$OUT/code_commit.txt"
  ENVX=(D=7 ROUNDS=7 SEEDS=0 UNITS="$U" NTRAIN=15000000
        EXTRA_POOL="${P7}_TRAINONLY_seed43.npz" EXTRA_N=5000000
        BATCH=10000 EPOCHS=200 LR="$LRV" LR_SCHEDULE="$SCH"
        EARLY_STOP=0 RESET_AFTER=0 UNROLL=0 EVAL_NO_TF32=1
        GPU_MEM_MIB="$CAP" MIN_FREE_MIB="$CAP" OUT="$OUT" STAMP="$S"
        RESUME_WEIGHTS="$OUT/resume_from/$TAGB.lastepoch.weights.h5" INITIAL_EPOCH=100
        RESUME_BEST_JSON="$OUT/seed0/ckpt/$TAGB.best.json"
        RESUME_HISTORY="$OUT/resume_from/$TAGB.history.json")
  echo "podman run --rm --name fd_$name --security-opt=label=disable --device nvidia.com/gpu=all -v $HOME:$HOME -e HOME=$HOME -w $REPO/src $IMG env ${ENVX[*]} bash gh200_run_student_scaling.sh" > "$OUT/command.txt"
  log "START $name (resume $SRCN from epoch 100 to 200) units=$U lr=$LRV -> $OUT (free GPU $(free_mib) MiB, host available $(host_avail_gb) GB, jobs running ${#PID[@]})"
  podman run --rm --name "fd_$name" --security-opt=label=disable --device nvidia.com/gpu=all \
     -v "$HOME:$HOME" -e HOME="$HOME" -w "$REPO/src" "$IMG" \
     env "${ENVX[@]}" bash gh200_run_student_scaling.sh > "$OUT/podman.log" 2>&1 9>&- &
  PID[$name]=$!; OUTD[$name]=$OUT; SCHED[$name]=$SCH
  for i in $(seq 1 60); do running "fd_$name" && break; sleep 5; done
}

alive () {    # alive <name>: is the job still going?
  local name="$1"
  if [ "${PID[$name]}" = adopted ]; then running "fd_$name"; else kill -0 "${PID[$name]}" 2>/dev/null; fi
}

clip_check () {
  local name="$1" SD f; SD="${OUTD[$name]}/seed0"
  [ "${SCHED[$name]}" = cosine ] && [ -f "$SD/run.log" ] || return 0
  f=$(grep -o 'cumulative [0-9.]*' "$SD/run.log" | tail -1 | cut -d' ' -f2)
  if [ -n "$f" ] && python3 -c "import sys; sys.exit(0 if float('$f') > $CLIP_LIMIT else 1)"; then
    podman stop -t 30 "fd_$name" >/dev/null
    echo "cumulative clipped fraction $f > $CLIP_LIMIT" > "${OUTD[$name]}/STOP_clipping_over_5pct"
    log "STOP  $name: cumulative clipped fraction $f > $CLIP_LIMIT"
  fi
}

finalize () {  # finalize <name>: exit code, DONE/FAIL line, summary
  local name="$1" OUT SD; OUT="${OUTD[$name]}"; SD="$OUT/seed0"
  if [ "${PID[$name]}" = adopted ]; then
    echo "unknown (podman process belonged to driver.sh; adopted by driver2.sh)" > "$OUT/exit_code"
  else
    wait "${PID[$name]}"; echo $? > "$OUT/exit_code"
  fi
  if [ ! -f "$SD/COMPLETE" ]; then
    log "FAIL  $name ended without COMPLETE (exit $(cat "$OUT/exit_code")); last lines: $(grep -E 'Error|error|STOP|Killed' "$SD/run.log" "$OUT/podman.log" 2>/dev/null | tail -2 | tr '\n' ' ')"
  elif "$PYH" "$DIR/summarize_run.py" "$OUT" "$RES" "fd_$name" > "$SD/summary.json" 2>> "$DIR/summarize.log"; then
    log "DONE  $name: $("$PYH" -c "import json;d=json.load(open('$SD/summary.json'));print(' '.join(f'{k}={d[k]}' for k in ('params','p_L','mwpm_p_L','ratio','best_epoch','best_val_loss','best_val_ler','final_train_loss','final_val_loss','runtime_h','peak_host_gb','peak_gpu_mib')))")"
  else
    log "DONE  $name (COMPLETE; summary failed, see summarize.log)"
  fi
  unset "PID[$name]" "OUTD[$name]" "SCHED[$name]"
}

log "BEGIN resume_driver (MAX_JOBS=$MAX_JOBS) pid=$$ repo=$(git -C "$REPO" rev-parse --short HEAD) free GPU $(free_mib) MiB host available $(host_avail_gb) GB"
NEXT=0
while :; do
  # skip queue entries that already have a run folder
  while [ "$NEXT" -lt "${#QUEUE[@]}" ] && ls -d "$DIR/${QUEUE[$NEXT]%% *}"_2* >/dev/null 2>&1; do NEXT=$((NEXT + 1)); done
  sample_resources
  for name in "${!PID[@]}"; do
    if alive "$name"; then clip_check "$name"; else sleep 5; finalize "$name"; fi
  done
  if [ "$NEXT" -lt "${#QUEUE[@]}" ] && [ "${#PID[@]}" -lt "$MAX_JOBS" ]; then
    need=$NEED_FREE_ALONE; [ "${#PID[@]}" -ge 1 ] && need=$NEED_FREE_SECOND
    fm=$(free_mib); hg=$(host_avail_gb)
    if [ "$fm" -ge "$need" ] && [ "$hg" -ge "$NEED_HOST_GB" ]; then
      launch ${QUEUE[$NEXT]}; NEXT=$((NEXT + 1))
      sleep 60   # let the new job allocate before the next free-memory check
    elif [ "${WAITLOGGED:-}" != "$NEXT" ]; then
      log "WAIT  next job needs free GPU >= $need MiB (now $fm) and host >= $NEED_HOST_GB GB (now $hg)"; WAITLOGGED=$NEXT
    fi
  fi
  [ "$NEXT" -ge "${#QUEUE[@]}" ] && [ "${#PID[@]}" -eq 0 ] && break
  sleep 30
done
log "END   resume queue finished"
