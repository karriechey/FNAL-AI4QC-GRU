#!/usr/bin/env bash
# Created: 2026-09-29
# Last updated: 2026-09-29
#
# Slot-based replacement for driver.sh (same queue, same settings, same per-run outputs).
# 2026-09-29 ~15:45Z: the shared GPU was freed (the other user's server stopped), so the
# job limit became a setting (MAX_JOBS, default 2) and the driver adopts every running
# fd_* job of this queue and skips queue entries that already have a folder. Restarted
# with MAX_JOBS=5 so F3-F5 run next to F1 and F2.
# driver.sh ran one job at a time; at ~16:00Z on 2026-09-29 it was stopped while F1 was
# training (F1's container kept running untouched) and this driver took over:
#
#   - it adopts the running F1 (clip check, DONE/FAIL line, summary.json when it ends;
#     F1's exit_code is not available because its podman process belonged to driver.sh)
#   - it starts F2 -> F3 -> F4 -> F5 in that order as slots free up
#   - at most MAX_JOBS training jobs at a time
#   - a job starts only if host memory available >= 150 GB and free GPU memory is
#     >= 2,500 MiB when another job is running (>= 2,000 MiB when none is); otherwise it
#     waits
#   - each job keeps the 1.6 GB GPU cap
#
# Job settings, logging (driver.log, resources.tsv, command.txt, podman.log, run.log,
# summary.json) and failure handling (FAIL line, queue continues) are as in driver.sh.
# The sealed partition [17M, 19M) is not read.
#
#     setsid nohup bash driver2.sh > /dev/null 2>&1 < /dev/null &
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
MAX_JOBS="${MAX_JOBS:-2}"
GCLIP=0.64
CLIP_LIMIT=0.05
P7="$HOME/pools_d7_p004_19M/data_d7_p0.004_r7"

# Earlier locks (driver.lock, driver2.lock) stay held by podman processes that inherited
# the descriptor, so this driver uses driver2b.lock and closes it for every podman it starts.
exec 9> "$DIR/driver2b.lock"
flock -n 9 || { echo "[foundational $(date -u '+%Y-%m-%dT%H:%M:%SZ')] another driver2 holds driver2b.lock; exiting" >> "$LOG"; exit 1; }

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
}

# Queue, in order: name units lr epochs schedule
QUEUE=("F2_d7_u70_20M 70 0.003 100 constant"
       "F3_d7_u200_20M 200 0.003 100 constant"
       "F4_d7_u140_20M_lr3e-4 140 0.0003 100 constant"
       "F5_d7_u140_20M_lr3e-5 140 0.00003 100 constant")
declare -A PID OUTD SCHED      # per active job: podman pid ("adopted" for F1), folder, schedule

launch () {   # launch <name> <units> <lr> <epochs> <schedule>
  local name="$1" U="$2" LRV="$3" EP="$4" SCH="$5" S OUT i ENVX
  S=$(date -u +%Y%m%dT%H%M%SZ); OUT="$DIR/${name}_$S"; mkdir -p "$OUT"
  { echo "at_start repo_head $(git -C "$REPO" rev-parse HEAD)"
    echo "at_start train_student_sha256 $(sha256sum "$REPO/src/train_student.py" | cut -d' ' -f1)"
  } > "$OUT/code_commit.txt"
  ENVX=(D=7 ROUNDS=7 SEEDS=0 UNITS="$U" NTRAIN=15000000
        EXTRA_POOL="${P7}_TRAINONLY_seed43.npz" EXTRA_N=5000000
        BATCH=10000 EPOCHS="$EP" LR="$LRV" LR_SCHEDULE="$SCH"
        EARLY_STOP=0 RESET_AFTER=0 UNROLL=0 EVAL_NO_TF32=1
        GPU_MEM_MIB="$CAP" MIN_FREE_MIB="$CAP" OUT="$OUT" STAMP="$S")
  [ "$SCH" = cosine ] && ENVX+=(LR_END=3e-5 GLOBAL_CLIPNORM="$GCLIP")
  echo "podman run --rm --name fd_$name --security-opt=label=disable --device nvidia.com/gpu=all -v $HOME:$HOME -e HOME=$HOME -w $REPO/src $IMG env ${ENVX[*]} bash gh200_run_student_scaling.sh" > "$OUT/command.txt"
  log "START $name d=7 units=$U lr=$LRV schedule=$SCH epochs=$EP -> $OUT (free GPU $(free_mib) MiB, host available $(host_avail_gb) GB, jobs running ${#PID[@]})"
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

log "BEGIN driver2 (MAX_JOBS=$MAX_JOBS) pid=$$ repo=$(git -C "$REPO" rev-parse --short HEAD) free GPU $(free_mib) MiB host available $(host_avail_gb) GB"
# Adopt every running job of this queue (its podman process belongs to an earlier driver).
for c in $(podman ps --format '{{.Names}}' | grep '^fd_F'); do
  name=${c#fd_}
  PID[$name]=adopted
  OUTD[$name]=$(ls -d "$DIR/${name}"_2*/ | tail -1 | sed 's:/$::')
  case "$name" in *_cos) SCHED[$name]=cosine ;; *) SCHED[$name]=constant ;; esac
  log "ADOPT $name running -> ${OUTD[$name]}"
done

NEXT=0
while :; do
  # skip queue entries that already have a run folder (started by an earlier driver)
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
log "END   foundational queue finished"
