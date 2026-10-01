#!/usr/bin/env bash
# Created: 2026-09-30
# Last updated: 2026-09-30
#
# Twelve-run queue 2026-09-30 (specified before any result). Every run uses the K1/L1
# recipe: 80 epochs, cosine LR from the run's peak to 3e-5 per update (no warmup), Adam
# global_clipnorm 0.64, batch 10,000, reset_after=False, seed 0, rounds r = d, early
# stopping off, determinism on, TF32 on for training / off for evaluation, best checkpoint
# by validation LER. Evaluation on the FORMAL pool's evaluation block [15.2M, 17M) with MWPM
# decoded on the same shots (eval_student_on_tail.py). The sealed partition is not read.
# The fraction of updates clipped is logged every epoch and reported per run (not a stop
# rule here).
#
# Jobs come from jobs.tsv, re-read every loop, so jobs can be appended when their pools are
# ready. A job starts when (all of):
#   - it has no run folder yet, and all its pool files and fingerprints exist;
#   - fewer than MAX_JOBS (5) jobs are running;
#   - no running job is still loading data (no "Epoch 1/" in its run.log yet), so large
#     loads happen one at a time;
#   - free GPU memory >= 2,500 MiB (>= 2,000 MiB when nothing of ours runs) and host memory
#     available >= 150 GB.
# Jobs are tried in file order; a job whose pools are missing is skipped for now, so later
# jobs can use a free slot.
#
# Divergence watch: a run whose training loss becomes nan is stopped; the epoch and the
# last finite loss are recorded (DIVERGED line, DIVERGED file) and the queue moves on.
#
# jobs.tsv columns (tab-separated; '#' lines ignored):
#   name d units peak_lr p formal_pool extra_pools(comma-separated) extra_counts(comma) gpu_cap_mib
#
# driver_b.sh (2026-09-30): the same driver for the runs that need the compact-pools code
# (runs 1-5, 8, 11, 12). It runs from the compact-pools worktree while driver.sh, in the main
# checkout, keeps managing runs 6, 7, 9, 10, whose files are not touched. Differences:
#   - REPO is the checkout this script sits in; files carry a _b suffix (driver_b.log,
#     jobs_b.tsv, resources_b.tsv, results_b.tsv) so the two folders merge without clashes;
#   - slots and the one-load-at-a-time rule count every q12_ container of both drivers;
#   - nothing starts until every job in the main checkout's jobs.tsv has started;
#   - optional 10th jobs column: host memory (GB) that must be available before that job
#     starts (default 150), set from its expected loading peak. If the next job whose pools
#     are ready does not have it, the driver waits instead of starting a later job.
#
# Changes 2026-10-01:
#   - optional 11th jobs column: epochs (default 80); run 13 uses 120;
#   - swap rule: a job starts only if swap in use has not grown by more than 2 GB over the
#     last 10 minutes (no active swapping) and host memory available >= its threshold.
#     Swap already in use from earlier (idle pages of running jobs) does not block by itself;
#   - job_env.txt (KEY=VALUE lines, re-read at every launch) adds environment settings to
#     every job started from then on (the memory settings chosen by the memory test);
#   - on start the driver adopts running q12_ jobs whose run folder is in this checkout
#     (their podman process belonged to an earlier copy of this driver).
#
#     setsid nohup bash driver_b.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="$REPO/experiments/queue12_2026-09-30"
MAINDIR="$HOME/FNAL-AI4QC-GRU/experiments/queue12_2026-09-30"
LOG="$DIR/driver_b.log"
RES="$DIR/resources_b.tsv"
JOBS="$DIR/jobs_b.tsv"
RESULTS="$DIR/results_b.tsv"
SUMMARIZE="$REPO/experiments/foundational_2026-09-29/summarize_run.py"   # same script in both checkouts
IMG=qdec:tf215
PYH="$HOME/tfenv/bin/python"
NEED_FREE_ALONE=2000
NEED_FREE_SECOND=2500
NEED_HOST_GB=150
MAX_JOBS="${MAX_JOBS:-5}"
GCLIP=0.64

exec 9> "$DIR/driver_b.lock"
flock -n 9 || { echo "[queue12 $(date -u '+%Y-%m-%dT%H:%M:%SZ')] another driver_b holds driver_b.lock; exiting" >> "$LOG"; exit 1; }

log () { echo "[queue12 $(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG"; }
free_mib () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
swap_used_gb () { awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{printf "%d", (t-f)/1048576}' /proc/meminfo; }
SWAPHIST=()   # swap in use at each loop (about every 30-60 s), last ~10 minutes
swap_growing () {  # has swap in use grown by > 2 GB over the samples kept?
  local now; now=$(swap_used_gb); SWAPHIST+=("$now")
  [ "${#SWAPHIST[@]}" -gt 15 ] && SWAPHIST=("${SWAPHIST[@]:1}")
  [ $(( now - SWAPHIST[0] )) -gt 2 ]
}
host_avail_gb () { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
running () { [ -n "$(podman ps -q --filter name="^$1\$")" ]; }
sample_resources () {
  local t c; t=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
  echo -e "$t\thost_available_gb\t$(host_avail_gb)" >> "$RES"
  podman stats --no-stream --format '{{.Name}}\t{{.MemUsage}}' 2>/dev/null | grep '^q12_' \
    | sed "s/^/$t\tcontainer_mem\t/" >> "$RES"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | sed "s/^/$t\tgpu_pid_mib\t/" >> "$RES"
  for c in $(podman ps --format '{{.Names}}' | grep '^q12_'); do
    echo -e "$t\tcontainer_hpids\t$c\t$(podman top "$c" hpid 2>/dev/null | tail -n +2 | tr '\n' ' ')" >> "$RES"
  done
}

declare -A PID=() OUTD=()

launch () {   # launch <name> <d> <units> <lr> <p> <pool> <extra pools csv> <extra counts csv> <gpu cap> <epochs>
  local name="$1" D="$2" U="$3" LRV="$4" PP="$5" POOLF="$6" XP="$7" XN="$8" CAP="$9" EP="${10:-80}" S OUT ENVX i kv
  S=$(date -u +%Y%m%dT%H%M%SZ); OUT="$DIR/${name}_$S"; mkdir -p "$OUT"
  { echo "at_start repo_head $(git -C "$REPO" rev-parse HEAD)"
    echo "at_start train_student_sha256 $(sha256sum "$REPO/src/train_student.py" | cut -d' ' -f1)"
  } > "$OUT/code_commit.txt"
  ENVX=(D="$D" ROUNDS="$D" SEEDS=0 UNITS="$U" P="$PP" POOL="$POOLF" NTRAIN=15000000
        BATCH=10000 EPOCHS="$EP" LR="$LRV" LR_SCHEDULE=cosine LR_END=3e-5 GLOBAL_CLIPNORM="$GCLIP"
        EARLY_STOP=0 RESET_AFTER=0 UNROLL=0 EVAL_NO_TF32=1
        GPU_MEM_MIB="$CAP" MIN_FREE_MIB="$CAP" OUT="$OUT" STAMP="$S")
  [ "$XP" != "-" ] && ENVX+=(EXTRA_POOL="${XP//,/ }" EXTRA_N="${XN//,/ }")
  if [ -f "$DIR/job_env.txt" ]; then
    while IFS= read -r kv; do [ -n "$kv" ] && [ "${kv:0:1}" != "#" ] && ENVX+=("$kv"); done < "$DIR/job_env.txt"
  fi
  echo "podman run --rm --name q12_$name --security-opt=label=disable --device nvidia.com/gpu=all -v $HOME:$HOME -e HOME=$HOME -w $REPO/src $IMG env ${ENVX[*]} bash gh200_run_student_scaling.sh" > "$OUT/command.txt"
  log "START $name d=$D units=$U peak_lr=$LRV p=$PP epochs=$EP extra=[$XN] env=[$(grep -v '^#' "$DIR/job_env.txt" 2>/dev/null | tr '\n' ' ')] -> $OUT (free GPU $(free_mib) MiB, host available $(host_avail_gb) GB, jobs running $(n_running))"
  podman run --rm --name "q12_$name" --security-opt=label=disable --device nvidia.com/gpu=all \
     -v "$HOME:$HOME" -e HOME="$HOME" -w "$REPO/src" "$IMG" \
     env "${ENVX[@]}" bash gh200_run_student_scaling.sh > "$OUT/podman.log" 2>&1 9>&- &
  PID[$name]=$!; OUTD[$name]=$OUT
  for i in $(seq 1 60); do running "q12_$name" && break; sleep 5; done
}

n_running () { podman ps --format '{{.Names}}' | grep -c '^q12_'; }

rundir () {   # latest run folder of a job, in this checkout or the main one
  ls -d "$DIR/${1}"_2* "$MAINDIR/${1}"_2* 2>/dev/null | tail -1
}

loading () {  # is any running q12_ job (either driver) still before its first epoch?
  local c f
  for c in $(podman ps --format '{{.Names}}' | grep '^q12_'); do
    f=$(rundir "${c#q12_}")
    grep -q 'Epoch 1/' "$f/seed0/run.log" 2>/dev/null || return 0
  done
  return 1
}

main_all_started () {  # every job of the main checkout's jobs.tsv has a run folder
  local name rest
  while IFS=$'\t' read -r name rest; do
    [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
    ls -d "$MAINDIR/${name}"_2* >/dev/null 2>&1 || return 1
  done < "$MAINDIR/jobs.tsv"
  return 0
}

nan_check () {
  local name="$1" SD="${OUTD[$1]}/seed0" last ep
  [ -f "$SD/run.log" ] || return 0
  if grep -qE 'loss: nan|loss=nan' "$SD/run.log"; then
    ep=$(grep -c 'ms/step' "$SD/run.log")
    last=$(grep -oE ' - loss: [0-9.e+-]+' "$SD/run.log" | tail -1 | awk '{print $3}')
    podman stop -t 30 "q12_$name" >/dev/null
    echo "training loss nan at epoch $ep; last finite epoch loss ${last:-none}" > "${OUTD[$name]}/DIVERGED"
    log "DIVERGED $name: loss nan at epoch $ep, last finite epoch loss ${last:-none}; stopped"
  fi
}

finalize () {
  local name="$1" OUT SD; OUT="${OUTD[$name]}"; SD="$OUT/seed0"
  if [ "${PID[$name]}" = adopted ]; then
    echo "unknown (podman process belonged to an earlier driver_b; adopted)" > "$OUT/exit_code"
  else
    wait "${PID[$name]}"; echo $? > "$OUT/exit_code"
  fi
  if [ ! -f "$SD/COMPLETE" ]; then
    log "FAIL  $name ended without COMPLETE (exit $(cat "$OUT/exit_code")); last lines: $(grep -E 'Error|error|STOP|Killed|DIVERGED' "$SD/run.log" "$OUT/podman.log" "$OUT/DIVERGED" 2>/dev/null | tail -2 | tr '\n' ' ')"
  elif "$PYH" "$SUMMARIZE" "$OUT" "$RES" "q12_$name" > "$SD/summary.json" 2>> "$DIR/summarize.log"; then
    log "DONE  $name: $("$PYH" -c "import json;d=json.load(open('$SD/summary.json'));print(' '.join(f'{k}={d.get(k)}' for k in ('params','p_L','mwpm_p_L','ratio','best_epoch','best_val_loss','best_val_ler','final_train_loss','final_val_loss','clip_frac_cum','runtime_h','peak_host_gb','peak_gpu_mib')))")"
    "$PYH" "$DIR/results_row.py" "$OUT" >> "$RESULTS" 2>> "$DIR/summarize.log"
  else
    log "DONE  $name (COMPLETE; summary failed, see summarize.log)"
  fi
  unset "PID[$name]" "OUTD[$name]"
}

log "BEGIN queue12 driver_b pid=$$ repo=$(git -C "$REPO" rev-parse --short HEAD) MAX_JOBS=$MAX_JOBS free GPU $(free_mib) MiB host available $(host_avail_gb) GB"
for c in $(podman ps --format '{{.Names}}' | grep '^q12_'); do
  f=$(ls -d "$DIR/${c#q12_}"_2* 2>/dev/null | tail -1)
  [ -n "$f" ] || continue
  PID[${c#q12_}]=adopted; OUTD[${c#q12_}]=$f
  log "ADOPT ${c#q12_} running -> $f"
done
IDLE=0
while :; do
  sample_resources
  for name in "${!PID[@]}"; do
    if { [ "${PID[$name]}" = adopted ] && running "q12_$name"; } || kill -0 "${PID[$name]}" 2>/dev/null; then nan_check "$name"; else sleep 5; finalize "$name"; fi
  done
  started=""
  NR=$(n_running)
  SG=no; swap_growing && SG=yes
  if [ "$SG" = yes ] && [ "${SWAPLOGGED:-}" != yes ]; then log "HOLD  swap in use grew by > 2 GB in the last ~10 min (now $(swap_used_gb) GB); no new starts"; SWAPLOGGED=yes; fi
  [ "$SG" = no ] && SWAPLOGGED=""
  if [ "$SG" = no ] && main_all_started && [ "$NR" -lt "$MAX_JOBS" ] && ! loading; then
    need=$NEED_FREE_ALONE; [ "$NR" -ge 1 ] && need=$NEED_FREE_SECOND
    fm=$(free_mib); hg=$(host_avail_gb)
    if [ "$fm" -ge "$need" ] && [ "$hg" -ge "$NEED_HOST_GB" ]; then
      while IFS=$'\t' read -r name d u lr pp pool xp xn cap hn ep; do
        [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
        ls -d "$DIR/${name}"_2* >/dev/null 2>&1 && continue          # started before
        [ -n "${PID[$name]:-}" ] && continue
        ok=1
        for f in "$pool" ${xp//,/ }; do
          [ "$f" = "-" ] && continue
          [ -f "$f" ] && [ -f "${f%.npz}.fingerprint.json" ] || ok=0
        done
        [ "$ok" = 1 ] || continue
        if [ "$hg" -lt "${hn:-$NEED_HOST_GB}" ]; then
          [ "${HOSTWAIT:-}" != "$name" ] && log "WAIT  $name needs host >= ${hn} GB available (now $hg)"; HOSTWAIT=$name; break
        fi
        launch "$name" "$d" "$u" "$lr" "$pp" "$pool" "$xp" "$xn" "$cap" "${ep:-80}"; started=1; break
      done < "$JOBS"
    fi
  fi
  if [ "$(n_running)" -eq 0 ] && [ -z "$started" ]; then
    IDLE=$((IDLE + 1))
    [ -f "$DIR/STOP_WHEN_IDLE" ] && [ "$IDLE" -ge 3 ] && break
  else
    IDLE=0
  fi
  [ -n "$started" ] && sleep 60 || sleep 30
done
log "END   queue12 driver_b (STOP_WHEN_IDLE present and nothing running or startable)"
