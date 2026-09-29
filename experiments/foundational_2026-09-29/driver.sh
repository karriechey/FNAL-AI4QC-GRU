#!/usr/bin/env bash
# Created: 2026-09-29
# Last updated: 2026-09-29
#
# Foundational queue 2026-09-29 (specified before any result), run unattended on grace1,
# one job at a time (one GPU; running two at once slows both).
#
# Fixed for every job: d=7, r=7, p=0.004, reset_after=False, seed 0, batch 10,000,
# training set 20M = FORMAL 15M + TRAINONLY seed43 5M, validation [15M, 15.2M),
# evaluation block [15.2M, 17M) with MWPM on the same shots, best checkpoint by validation
# LER, early stopping off, determinism on, TF32 on for training / off for evaluation,
# 1.6 GB GPU cap. The sealed partition [17M, 19M) is not read.
#
#   F1  u280  cosine 3e-3 -> 3e-5, global_clipnorm 0.64, 80 epochs  (completes the
#       20M -> 50M (K1) -> 100M (L1) ladder under the same recipe)
#   F2  u70   constant 3e-3, 100 epochs
#   F3  u200  constant 3e-3, 100 epochs
#   F4  u140  constant 3e-4, 100 epochs
#   F5  u140  constant 3e-5, 100 epochs
#
# A failed job is recorded (FAIL line) and the queue moves on. Every 30 s resources.tsv gets
# host memory available, each container's memory (podman stats) and each process's GPU
# memory. After each job, summarize_run.py writes <run>/seed0/summary.json.
# A lock file stops a second copy of this driver from running.
#
#     setsid nohup bash driver.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail

REPO="$HOME/FNAL-AI4QC-GRU"
DIR="$REPO/experiments/foundational_2026-09-29"
LOG="$DIR/driver.log"
RES="$DIR/resources.tsv"
IMG=qdec:tf215
PYH="$HOME/tfenv/bin/python"
CAP=1600
NEED_FREE=2000
GCLIP=0.64
CLIP_LIMIT=0.05
P7="$HOME/pools_d7_p004_19M/data_d7_p0.004_r7"
mkdir -p "$DIR"

exec 9> "$DIR/driver.lock"
flock -n 9 || { echo "[foundational $(date -u '+%Y-%m-%dT%H:%M:%SZ')] another driver holds driver.lock; exiting" >> "$LOG"; exit 1; }

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

run_job () {  # run_job <name> <units> <lr> <epochs> <schedule: constant|cosine>
  local name="$1" U="$2" LRV="$3" EP="$4" SCH="$5" S OUT SD f i ENVX
  until [ "$(free_mib)" -ge "$NEED_FREE" ]; do sleep 60; done
  S=$(date -u +%Y%m%dT%H%M%SZ); OUT="$DIR/${name}_$S"; SD="$OUT/seed0"; mkdir -p "$OUT"
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
  log "START $name d=7 units=$U lr=$LRV schedule=$SCH epochs=$EP -> $OUT (free GPU $(free_mib) MiB, host available $(host_avail_gb) GB)"
  podman run --rm --name "fd_$name" --security-opt=label=disable --device nvidia.com/gpu=all \
     -v "$HOME:$HOME" -e HOME="$HOME" -w "$REPO/src" "$IMG" \
     env "${ENVX[@]}" bash gh200_run_student_scaling.sh > "$OUT/podman.log" 2>&1 &
  local PODPID=$!
  for i in $(seq 1 60); do running "fd_$name" && break; sleep 5; done
  while kill -0 "$PODPID" 2>/dev/null; do
    sample_resources
    if [ "$SCH" = cosine ] && [ -f "$SD/run.log" ]; then
      f=$(grep -o 'cumulative [0-9.]*' "$SD/run.log" | tail -1 | cut -d' ' -f2)
      if [ -n "$f" ] && python3 -c "import sys; sys.exit(0 if float('$f') > $CLIP_LIMIT else 1)"; then
        podman stop -t 30 "fd_$name" >/dev/null
        echo "cumulative clipped fraction $f > $CLIP_LIMIT" > "$OUT/STOP_clipping_over_5pct"
        log "STOP  $name: cumulative clipped fraction $f > $CLIP_LIMIT"
      fi
    fi
    sleep 30
  done
  wait "$PODPID"; echo $? > "$OUT/exit_code"
  if [ ! -f "$SD/COMPLETE" ]; then
    log "FAIL  $name ended without COMPLETE (exit $(cat "$OUT/exit_code")); last lines: $(grep -E 'Error|error|STOP|Killed' "$SD/run.log" "$OUT/podman.log" 2>/dev/null | tail -2 | tr '\n' ' ')"
    return
  fi
  "$PYH" "$DIR/summarize_run.py" "$OUT" "$RES" "fd_$name" > "$SD/summary.json" 2>> "$DIR/summarize.log" \
    && log "DONE  $name: $("$PYH" -c "import json;d=json.load(open('$SD/summary.json'));print(' '.join(f'{k}={d[k]}' for k in ('params','p_L','mwpm_p_L','ratio','best_epoch','best_val_loss','best_val_ler','final_train_loss','final_val_loss','runtime_h','peak_host_gb','peak_gpu_mib')))")" \
    || log "DONE  $name (COMPLETE; summary failed, see summarize.log)"
}

log "BEGIN foundational driver pid=$$ repo=$(git -C "$REPO" rev-parse --short HEAD) free GPU $(free_mib) MiB host available $(host_avail_gb) GB"
run_job F1_d7_u280_20M_cos 280 0.003  80 cosine
run_job F2_d7_u70_20M      70  0.003  100 constant
run_job F3_d7_u200_20M     200 0.003  100 constant
run_job F4_d7_u140_20M_lr3e-4 140 0.0003 100 constant
run_job F5_d7_u140_20M_lr3e-5 140 0.00003 100 constant
log "END   foundational queue finished"
