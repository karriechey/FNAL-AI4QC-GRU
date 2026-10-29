#!/usr/bin/env bash
# Created: 2026-10-28
# Last modified: 2026-10-28
# One run of PLAN.md by name, resumable. Waits for GPU, host memory and disk; trains with an
# epoch snapshot (weights + Adam) every epoch, kept in $SNAP_ROOT because /home is nearly full;
# applies the kill rule; then (L sweep) scores the selection block, selects the checkpoint and
# scores it on the eval block. Run again after a crash and it continues from the last epoch.
#   bash run_dvd.sh lrscan_d5_L01_lr1e-3
#   bash run_dvd.sh dvd_d7_L03_u168          (needs lr/d7.json)
set -uo pipefail
RUN=$1
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../../src" && pwd)"
SEL="$(cd "$HERE/../selection_2026-10-14" && pwd)"
KD="$HERE/../kd_pilot_p004_2026-10"
DEPTH="$HERE/../depth_scaling_d7_2026-10"
HOSTPY=${HOSTPY:-$HOME/tfenv/bin/python}
STATUS="$HERE/status.log"
SNAP_ROOT=${SNAP_ROOT:-/tmp/kchey_dvd}
MAX_CONCURRENT=${MAX_CONCURRENT:-8}       # 72 cores, 8 sampling workers per run
MIN_FREE_HOME_GB=${MIN_FREE_HOME_GB:-1}
MIN_FREE_SNAP_GB=${MIN_FREE_SNAP_GB:-5}
# other users' GPU memory is headroom lost, not a reason to wait; an idle check runs instead
FOREIGN_MAX_MIB=${FOREIGN_MAX_MIB:-100000}

status () { echo "$(date -u +%FT%TZ) $RUN $*" >> "$STATUS"; }
report () { flock "$HERE/.report.lock" "$HOSTPY" "$HERE/dvd.py" report > /dev/null 2>&1; }
mkdir -p "$HERE/runs" "$SNAP_ROOT"

eval "$("$HOSTPY" "$HERE/dvd.py" info "$RUN")" || exit 1
[ -n "$LRV" ] || { echo "no peak LR for d$D yet (lr/d$D.json)"; exit 1; }
CN="dvd_$RUN"
podman ps --format '{{.Names}}' | grep -qx "$CN" && { echo "$RUN is running"; exit 0; }

UNITS=$(printf "$W %.0s" $(seq "$L")); UNITS=${UNITS% }
GOT=$(cd "$SRC" && CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 "$HOSTPY" -c "
import sys
from StudentModels import build_gru_student
u = [int(x) for x in sys.argv[1].split()]
print(build_gru_student(d=$D, rounds=$D, inputs='evts', units=u[0] if len(u) == 1 else tuple(u), hidden=(), p=0.004).count_params())
" "$UNITS" 2>/dev/null | tail -1)
[ "$GOT" = "$WANT" ] || { status "FAILED: parameter assertion, units ($UNITS) count_params $GOT, plan $WANT"; exit 1; }
TF32=$(cat "$HOME/FNAL-AI4QC-GRU-kd/experiments/kd_pilot_p004_2026-10/teacher_check/d$D/tf32_decision.txt" 2>/dev/null)
[ "$TF32" = on ] || [ "$TF32" = off ] || { status "FAILED: no kd pilot d$D TF32 decision"; exit 1; }
POOL=$(ls "$HOME"/pools_d${D}_p004_19M/data_d${D}_p0.004_r${D}_FORMAL.npz)

# where to pick up: a finished folder is left alone, an unfinished one is resumed
OUT=$(ls -d "$HERE"/runs/${RUN}_2* 2>/dev/null | tail -1)
DONE_EP=0
if [ -n "$OUT" ]; then
  for m in DONE KILLED DIVERGED; do [ -f "$OUT/$m" ] && { echo "$RUN already $m"; exit 0; }; done
  rm -f "$OUT/FAILED"
  SD="$OUT/seed$SEED"
  if [ ! -f "$SD/COMPLETE" ] && [ ! -f "$OUT/COMPLETE" ]; then
    if [ -s "$SD/run.log" ]; then
      n=1; while [ -e "$SD/run_part$n.log" ]; do n=$((n + 1)); done
      prev=$((n - 1))
      { [ "$prev" -ge 1 ] && cmp -s "$SD/run.log" "$SD/run_part$prev.log"; } || cp "$SD/run.log" "$SD/run_part$n.log"
    fi
    DONE_EP=$("$HOSTPY" -c "
import sys; sys.path.insert(0, '$HERE')
import dvd
ep = dvd.epochs_of(sys.argv[1])
print(max(ep) if ep else 0)" "$OUT")
    LAST_W=$(printf '%s/epochs/epoch%03d.weights.h5' "$SD" "$DONE_EP")
    if [ "$DONE_EP" -gt 0 ] && [ -s "$LAST_W" ] && CUDA_VISIBLE_DEVICES= "$HOSTPY" -c "
import h5py, sys
with h5py.File(sys.argv[1]) as h:
    assert 'optimizer' in h
" "$LAST_W" 2>/dev/null; then
      LOGARGS=(); for f in "$SD"/run_part*.log; do LOGARGS+=(--log "$f"); done
      "$HOSTPY" "$DEPTH/history_from_log.py" "${LOGARGS[@]}" --epochs "$DONE_EP" --out "$SD/resume_history.json" \
        || { status "FAILED: resume history"; touch "$OUT/FAILED"; exit 1; }
    else
      status "no usable snapshot in $(basename "$OUT") (epoch $DONE_EP); starting a new folder"
      OUT=""; DONE_EP=0
    fi
  fi
fi
if [ -z "$OUT" ]; then
  S=$(date -u +%Y%m%dT%H%M%SZ); OUT="$HERE/runs/${RUN}_$S"; mkdir -p "$OUT/seed$SEED"
  mkdir -p "$SNAP_ROOT/${RUN}_$S"
  ln -sfn "$SNAP_ROOT/${RUN}_$S" "$OUT/seed$SEED/epochs"
fi
S=${OUT##*_}; SD="$OUT/seed$SEED"
[ -d "$SD/epochs/" ] || { status "FAILED: snapshot folder $(readlink "$SD/epochs") missing"; touch "$OUT/FAILED"; exit 1; }

pod () {   # container name, command...
  local name=$1; shift
  podman run --rm --name "$name" --security-opt=label=disable --device nvidia.com/gpu=all \
    -v "$HOME:$HOME" -v "$SNAP_ROOT:$SNAP_ROOT" -e HOME="$HOME" \
    -e TF_DETERMINISTIC_OPS=1 -e TF_CUDNN_DETERMINISTIC=1 -e MALLOC_ARENA_MAX=2 "$@"
}

wait_for_room () {   # as run_depth.sh, plus the snapshot disk
  local waited="" why free_gb snap_gb avail_gb ours gpu_free foreign foreign_mib pid mib u
  while true; do
    [ -f "$HERE/STOP" ] && { status "not started: STOP file present"; exit 1; }
    free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
    snap_gb=$(df -BG --output=avail "$SNAP_ROOT" | tail -1 | tr -dc 0-9)
    avail_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
    ours=$(podman ps --format '{{.Names}}' | grep -c '^dvd')
    gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    foreign_mib=0; foreign=""
    while IFS=, read -r pid mib; do
      [ -n "$pid" ] || continue
      u=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
      [ "$u" = "$USER" ] || [ -z "$u" ] && continue
      foreign_mib=$((foreign_mib + ${mib%.*})); foreign="$foreign $pid($u,${mib}MiB)"
    done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr -d ' ')
    why=""
    [ "$free_gb" -ge "$MIN_FREE_HOME_GB" ] || why="only ${free_gb} GB free on /home"
    [ -n "$why" ] || [ "$snap_gb" -ge "$MIN_FREE_SNAP_GB" ] || why="only ${snap_gb} GB free on $SNAP_ROOT"
    [ -n "$why" ] || [ "$avail_gb" -ge "$NEED_HOST_GB" ] || why="host MemAvailable ${avail_gb} GB < $NEED_HOST_GB"
    [ -n "$why" ] || [ "$ours" -lt "$MAX_CONCURRENT" ] || why="$ours containers up (max $MAX_CONCURRENT)"
    [ -n "$why" ] || [ "${gpu_free:-0}" -ge $((GPU_MEM_MIB + 2048)) ] || why="GPU free ${gpu_free} MiB < $((GPU_MEM_MIB + 2048))"
    [ -n "$why" ] || [ "$foreign_mib" -lt "$FOREIGN_MAX_MIB" ] || why="other users on the GPU:$foreign"
    if [ -z "$why" ] && [ -n "$foreign" ] && [ "$ours" = 0 ]; then
      local s=(); for _ in $(seq 30); do s+=($(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)); sleep 10; done
      local mean peak
      mean=$(printf '%s\n' "${s[@]}" | awk '{t += $1} END {print t / NR}')
      peak=$(printf '%s\n' "${s[@]}" | sort -n | tail -1)
      awk -v m="$mean" -v p="$peak" 'BEGIN {exit !(m < 5 && p < 20)}' || why="GPU busy:$foreign (5 min mean $mean%, max $peak%)"
    fi
    [ -z "$why" ] && return
    [ "$why" = "$waited" ] || status "waiting: $why"
    waited=$why
    sleep 300
  done
}

prune_snapshots () {   # the two newest always; the L sweep also keeps epoch 20 and 40 on for selection scoring
  local files keep n e
  mapfile -t files < <(ls "$SD"/epochs/epoch*.weights.h5 2>/dev/null | sort)
  n=${#files[@]}
  for ((i = 0; i < n - 2; i++)); do
    e=$((10#$(basename "${files[$i]}" .weights.h5 | sed 's/epoch//')))
    keep=0
    [ "$KIND" = sweep ] && { [ "$e" = 20 ] || [ "$e" -ge 40 ]; } && keep=1
    [ $keep = 1 ] || rm -f "${files[$i]}"
  done
}

ENVX=(D=$D ROUNDS=$D SEEDS=$SEED UNITS="$UNITS" P=0.004 POOL=$POOL
      NTRAIN=0 FRESH_SHOTS=100000000 FRESH_WORKERS=8 FRESH_SEED_BASE=$FSB
      BATCH=10000 EPOCHS=$EPOCHS LR=$LRV LR_SCHEDULE=cosine LR_END=3e-5 LR_WARMUP_EPOCHS=3 GLOBAL_CLIPNORM=0.64
      EARLY_STOP=0 RESET_AFTER=0 UNROLL=0 EVAL_NO_TF32=1 TRAIN_NO_TF32=$([ "$TF32" = on ] && echo 0 || echo 1)
      MALLOC_ARENA_MAX=2 ALPHA=1.0 TEMPERATURE=1.0 SKIP_TAIL_EVAL=1 SAVE_EPOCHS_FROM=1
      OUT="$OUT" STAMP="$S")
[ -n "$STOPS" ] && ENVX+=(STOP_AFTER_STEPS=$STOPS)

if [ ! -f "$SD/COMPLETE" ] && [ ! -f "$OUT/COMPLETE" ]; then
  for attempt in 1 2; do
    wait_for_room
    RES=()
    if [ "$DONE_EP" -gt 0 ]; then
      RES=(RESUME_WEIGHTS="$(printf '%s/epochs/epoch%03d.weights.h5' "$SD" "$DONE_EP")" INITIAL_EPOCH=$DONE_EP
           RESUME_HISTORY="$SD/resume_history.json")
      BEST_J=$(ls "$SD"/ckpt/*.best.json 2>/dev/null | head -1)
      [ -n "$BEST_J" ] && RES+=(RESUME_BEST_JSON="$BEST_J")
    fi
    { echo "code $(head -1 "$HERE/../../CODE_COMMIT.txt" 2>/dev/null)"
      echo "run $RUN part $PART d=$D L=$L width $W seed $SEED params $GOT peak_lr $LRV tf32_train $TF32"
      printf '%s\n' "${ENVX[@]}" "GPU_MEM_MIB=$GPU_MEM_MIB" "${RES[@]}"; } > "$OUT/env.txt"
    status "started d=$D L=$L u$W seed=$SEED lr=$LRV params=$GOT gpu_mib=$GPU_MEM_MIB${RES:+ resumed from epoch $DONE_EP} -> $(basename "$OUT")"
    pod "$CN" -w "$SRC" qdec:tf215 env "${ENVX[@]}" GPU_MEM_MIB=$GPU_MEM_MIB MIN_FREE_MIB=$GPU_MEM_MIB "${RES[@]}" \
      bash gh200_run_student_scaling.sh > "$OUT/podman.log" 2>&1 &
    PODPID=$!
    last=0
    while kill -0 $PODPID 2>/dev/null; do
      sleep 60
      "$HOSTPY" "$HERE/dvd.py" kill-check "$OUT" > "$OUT/kill_check.txt" 2>&1
      rc=$?
      ep=$("$HOSTPY" -c "import sys; sys.path.insert(0, '$HERE'); import dvd; e = dvd.epochs_of(sys.argv[1]); print(max(e) if e else 0)" "$OUT" 2>/dev/null)
      if [ "${ep:-0}" -gt "$last" ]; then status "epoch $ep"; last=$ep; prune_snapshots; fi
      if [ $rc = 3 ]; then
        podman stop -t 30 "$CN" > /dev/null 2>&1
        wait $PODPID
        status "killed: $(cat "$OUT"/KILLED "$OUT"/DIVERGED 2>/dev/null)"
        report; exit 0
      fi
    done
    wait $PODPID; rc=$?
    { [ -f "$SD/COMPLETE" ] || [ -f "$OUT/COMPLETE" ]; } && break
    # out of GPU memory: once more with twice the cap, from the last snapshot
    if grep -qE 'ResourceExhausted|OOM when allocating|out of memory' "$OUT/podman.log" "$SD/run.log" 2>/dev/null && [ $attempt = 1 ]; then
      GPU_MEM_MIB=$((GPU_MEM_MIB * 2))
      status "GPU out of memory, retrying with gpu_mib=$GPU_MEM_MIB"
      n=1; while [ -e "$SD/run_part$n.log" ]; do n=$((n + 1)); done
      cp "$SD/run.log" "$SD/run_part$n.log" 2>/dev/null
      DONE_EP=$("$HOSTPY" -c "import sys; sys.path.insert(0, '$HERE'); import dvd; e = dvd.epochs_of(sys.argv[1]); print(max(e) if e else 0)" "$OUT")
      if [ "$DONE_EP" -gt 0 ] && [ -s "$(printf '%s/epochs/epoch%03d.weights.h5' "$SD" "$DONE_EP")" ]; then
        LOGARGS=(); for f in "$SD"/run_part*.log; do LOGARGS+=(--log "$f"); done
        "$HOSTPY" "$DEPTH/history_from_log.py" "${LOGARGS[@]}" --epochs "$DONE_EP" --out "$SD/resume_history.json" || DONE_EP=0
      else
        DONE_EP=0
      fi
      continue
    fi
    status "FAILED: training exit $rc, no COMPLETE (see $(basename "$OUT")/podman.log)"
    touch "$OUT/FAILED"; report; exit 1
  done
  status "trained"
fi
prune_snapshots
if [ "$KIND" = scan ]; then touch "$OUT/DONE"; status "done"; report; exit 0; fi

W_LIST=(); for e in 20 $(seq 40 "$EPOCHS"); do W_LIST+=("$SD/epochs/epoch$(printf '%03d' "$e").weights.h5"); done
for w in "${W_LIST[@]}"; do [ -s "$w" ] || { status "FAILED: missing snapshot $w"; touch "$OUT/FAILED"; report; exit 1; }; done
read -r -a UL <<< "$UNITS"
SCORE_MIB=$(( GPU_MEM_MIB > 8000 ? GPU_MEM_MIB : 8000 ))
pod "dvdsel_$RUN" -w "$SEL" qdec:tf215 python3 score_selection.py \
  --model "$RUN" --d $D --p 0.004 --units "${UL[@]}" --pool "$POOL" --gpu-mem-mib $SCORE_MIB \
  --steps-per-epoch 10000 --out-csv "$OUT/selection_scores.csv" "${W_LIST[@]}" > "$OUT/selection.log" 2>&1
status "selection-block scores exit $?"
"$HOSTPY" "$KD/select_checkpoint.py" --model "$RUN" --csv "$OUT/selection_scores.csv" --min-epoch 40 --epochs "$EPOCHS" \
  --weights-dir "$SD/epochs" --out-json "$OUT/selected.json" >> "$OUT/selection.log" 2>&1 \
  || { status "FAILED: checkpoint selection, see selection.log"; touch "$OUT/FAILED"; report; exit 1; }
SELW=$("$HOSTPY" -c "import json,sys; print(json.load(open(sys.argv[1]))['weights'])" "$OUT/selected.json")
cp "$SELW" "$OUT/selected.weights.h5"
pod "dvdeval_$RUN" -w "$SRC" qdec:tf215 python3 eval_student_on_tail.py --student gru --inputs evts --hidden \
  --units "${UL[@]}" --d $D --p 0.004 --rounds $D --pool "$POOL" --weights "$OUT/selected.weights.h5" \
  --eval-start $EVAL_START --n-test $N_EVAL --batch-size 10000 --no-tf32 --gpu-mem-mib $SCORE_MIB \
  --dump-per-shot "$OUT/eval_selected_per_shot.npz" --out-csv "$OUT/eval_selected.csv" > "$OUT/eval_selected.log" 2>&1
rc=$?
status "eval block ($(basename "$SELW")) exit $rc"
if [ $rc = 0 ]; then touch "$OUT/DONE"; status "done"; else touch "$OUT/FAILED"; fi
report
