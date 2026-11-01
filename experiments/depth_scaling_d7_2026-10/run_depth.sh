#!/usr/bin/env bash
# Created: 2026-10-21
# Last modified: 2026-10-23 (RESUME_OUT: continue a run the 2026-10-22 reboot killed)
# One run of prereg/depth_scaling_d7_2026-10.md by name. Waits for GPU, host memory and disk
# before starting, applies the kill rule while training, then (depth sweep) scores the
# selection block, selects the checkpoint and scores it on the eval block.
#   bash run_depth.sh lrscan_d7_L10_lr1e-3
#   bash run_depth.sh depth_d7_L04_u70        (needs depth_sweep/sweep_lr.txt)
#   SMOKE=1 bash run_depth.sh depth_d7_L10_u70   (2 epochs of 10M shots, for timing)
#   RESUME_OUT=<run folder> bash run_depth.sh depth_d7_L10_u70   (continue into that folder)
set -uo pipefail
RUN=$1
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../../src" && pwd)"
KD="$HERE/../kd_pilot_p004_2026-10"
HOSTPY=${HOSTPY:-$HOME/tfenv/bin/python}
STATUS="$HERE/status.log"
# kd pilot d7 TF32 decision (teacher check on grace1); scoring is always TF32 off
TF32_FILE=${TF32_FILE:-$HOME/FNAL-AI4QC-GRU-kd/experiments/kd_pilot_p004_2026-10/teacher_check/d7/tf32_decision.txt}
POOL=$HOME/pools_d7_p004_19M/data_d7_p0.004_r7_FORMAL.npz
FSB=4000000000000
# from the L=10 smoke (README, Decisions during the run): training peak 4.7 GB GPU, 4.6 GB host;
# selection scoring 20 GB host
GPU_MEM_MIB=${GPU_MEM_MIB:-6000}
NEED_HOST_GB=${NEED_HOST_GB:-30}
MAX_CONCURRENT=${MAX_CONCURRENT:-12}
MIN_FREE_GB=${MIN_FREE_GB:-3}
# memory other users may hold on the card before this run waits for them
FOREIGN_MAX_MIB=${FOREIGN_MAX_MIB:-10240}

status () { echo "$(date -u +%FT%TZ) $RUN $*" >> "$STATUS"; }
stop_all () {    # protocol stop: reason goes to the top of status.log
  { echo "STOP $(date -u +%FT%TZ) $RUN: $*"; cat "$STATUS" 2>/dev/null; } > "$STATUS.tmp" && mv "$STATUS.tmp" "$STATUS"
  touch "$HERE/STOP"
  exit 1
}

case "$RUN" in
  lrscan_d7_L[0-9][0-9]_lr*)
    PART=A; L=$((10#${RUN:11:2})); W=70; LRN=${RUN#*_lr}
    case "$LRN" in 5e-4|1e-3|2e-3|3e-3|5e-3) ;; *) echo "LR $LRN not in the prereg"; exit 1 ;; esac
    LRV=$LRN ;;
  depth_d7_L[0-9][0-9]_u70)
    PART=B; L=$((10#${RUN:10:2})); W=70 ;;
  match_d7_L04_u172) PART=B; L=1; W=172 ;;
  match_d7_L08_u255) PART=B; L=1; W=255 ;;
  *) echo "unknown run $RUN (README.md lists the runs)"; exit 1 ;;
esac
[ "$L" -ge 1 ] && [ "$L" -le 10 ] || { echo "L=$L outside 1-10"; exit 1; }
if [ "$PART" = B ]; then
  if [ -n "${SMOKE:-}" ]; then LRV=1e-3
  else
    [ -f "$HERE/depth_sweep/sweep_lr.txt" ] || { echo "no depth_sweep/sweep_lr.txt (Part A not finished). STOP."; exit 1; }
    LRV=$(cat "$HERE/depth_sweep/sweep_lr.txt")
  fi
fi
UNITS=$(printf "$W %.0s" $(seq "$L")); UNITS=${UNITS% }

case "$W" in    # prereg parameter table
  70) WANT=$((25061 + 29610 * (L - 1))) ;;
  172) WANT=114209 ;;
  255) WANT=232816 ;;
esac
GOT=$(cd "$SRC" && CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 "$HOSTPY" -c "
import sys
from StudentModels import build_gru_student
u = [int(x) for x in sys.argv[1].split()]
print(build_gru_student(d=7, rounds=7, inputs='evts', units=u[0] if len(u) == 1 else tuple(u), hidden=(), p=0.004).count_params())
" "$UNITS" 2>/dev/null | tail -1)
[ "$GOT" = "$WANT" ] || stop_all "parameter assertion failed: units ($UNITS) count_params $GOT, prereg $WANT"

TF32=$(cat "$TF32_FILE" 2>/dev/null)
[ "$TF32" = on ] || [ "$TF32" = off ] || stop_all "no kd pilot d7 TF32 decision at $TF32_FILE"

EPOCHS=80; FRESH=100000000; WU=3
if [ "$PART" = A ]; then BASE="$HERE/lr_scan"; STOPS=200000; SAVE_FROM=""
else BASE="$HERE/depth_sweep"; STOPS=""; SAVE_FROM=20; fi
if [ -n "${SMOKE:-}" ]; then
  FRESH=10000000; EPOCHS=2; WU=0; BASE="$HERE/smoke"
  if [ "$PART" = A ]; then STOPS=1000; else SAVE_FROM=1; fi
fi
SEL_CSV="$BASE/selection_scores.csv"
mkdir -p "$BASE"

# wait for room: disk, host memory, concurrency, GPU memory, other users on the GPU
waited=""
while true; do
  [ -f "$HERE/STOP" ] && { status "not started: STOP file present"; exit 1; }
  free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
  avail_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  ours=$(podman ps --format '{{.Names}}' | grep -c '^dep_')
  gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  foreign_mib=0; foreign=""
  while IFS=, read -r pid mib; do
    [ -n "$pid" ] || continue
    u=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    [ "$u" = "$USER" ] || [ -z "$u" ] && continue
    foreign_mib=$((foreign_mib + ${mib%.*})); foreign="$foreign $pid($u,${mib}MiB)"
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr -d ' ')
  why=""
  [ "$free_gb" -ge "$MIN_FREE_GB" ] || why="only ${free_gb} GB free on /home (need $MIN_FREE_GB)"
  [ -n "$why" ] || [ "$avail_gb" -ge "$NEED_HOST_GB" ] || why="host MemAvailable ${avail_gb} GB < $NEED_HOST_GB"
  [ -n "$why" ] || [ "$ours" -lt "$MAX_CONCURRENT" ] || why="$ours runs up (max $MAX_CONCURRENT)"
  [ -n "$why" ] || [ "${gpu_free:-0}" -ge $((GPU_MEM_MIB + 2048)) ] || why="GPU free ${gpu_free} MiB < $((GPU_MEM_MIB + 2048))"
  [ -n "$why" ] || [ "$foreign_mib" -lt "$FOREIGN_MAX_MIB" ] || why="other users on the GPU:$foreign"
  if [ -z "$why" ] && [ -n "$foreign" ] && [ "$ours" = 0 ]; then
    # as experiments/mwpm_fixes_d9_2026-10/driver.py: 5 minutes near 0% before sharing
    u=(); for _ in $(seq 30); do u+=($(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)); sleep 10; done
    mean=$(printf '%s\n' "${u[@]}" | awk '{s += $1} END {print s / NR}')
    peak=$(printf '%s\n' "${u[@]}" | sort -n | tail -1)
    awk -v m="$mean" -v p="$peak" 'BEGIN {exit !(m < 5 && p < 20)}' || why="GPU busy:$foreign (5 min mean $mean%, max $peak%)"
  fi
  [ -z "$why" ] && break
  [ "$why" = "$waited" ] || status "waiting: $why"
  waited=$why
  sleep 300
done

if [ -n "${RESUME_OUT:-}" ]; then
  OUT="$RESUME_OUT"; S=${OUT##*_}
  [ -d "$OUT/seed0/epochs" ] || { echo "no $OUT/seed0/epochs to resume from"; exit 1; }
  [ -f "$OUT/seed0/COMPLETE" ] && { echo "$OUT already COMPLETE"; exit 1; }
  # the epoch snapshots carry the Adam slots too, and one of them pairs with the last
  # complete epoch line in run.log; .lastepoch can be a partial write from the reboot
  DONE_EP=$(grep -cE '^[0-9]+/[0-9]+ - [0-9]+s - loss' "$OUT/seed0/run.log")
  LAST_W=$(printf '%s/seed0/epochs/epoch%03d.weights.h5' "$OUT" "$DONE_EP")
  BEST_J=$(ls "$OUT"/seed0/ckpt/*.best.json 2>/dev/null | head -1)
  [ -s "$LAST_W" ] || { echo "no epoch $DONE_EP snapshot at $LAST_W"; exit 1; }
  CUDA_VISIBLE_DEVICES= "$HOSTPY" -c "
import h5py, sys
with h5py.File(sys.argv[1]) as h:
    assert 'optimizer' in h, 'no optimizer state'
" "$LAST_W" || { echo "$LAST_W unreadable or without Adam state"; exit 1; }
  # the launcher truncates run.log, and train_student needs the finished epochs as history
  [ -f "$OUT/seed0/run_part1.log" ] || cp "$OUT/seed0/run.log" "$OUT/seed0/run_part1.log"
  "$HOSTPY" "$HERE/history_from_log.py" --log "$OUT/seed0/run_part1.log" --epochs "$DONE_EP" \
    --out "$OUT/seed0/resume_history.json" || exit 1
else
  S=$(date -u +%Y%m%dT%H%M%SZ); OUT="$BASE/${RUN}_$S"; mkdir -p "$OUT"
fi
ENVX=(D=7 ROUNDS=7 SEEDS=0 UNITS="$UNITS" P=0.004 POOL=$POOL
      NTRAIN=0 FRESH_SHOTS=$FRESH FRESH_WORKERS=8 FRESH_SEED_BASE=$FSB
      BATCH=10000 EPOCHS=$EPOCHS LR=$LRV LR_SCHEDULE=cosine LR_END=3e-5 LR_WARMUP_EPOCHS=$WU GLOBAL_CLIPNORM=0.64
      EARLY_STOP=0 RESET_AFTER=0 UNROLL=0 EVAL_NO_TF32=1 TRAIN_NO_TF32=$([ "$TF32" = on ] && echo 0 || echo 1)
      GPU_MEM_MIB=$GPU_MEM_MIB MIN_FREE_MIB=$GPU_MEM_MIB OUT="$OUT" STAMP="$S" MALLOC_ARENA_MAX=2
      ALPHA=1.0 TEMPERATURE=1.0 SKIP_TAIL_EVAL=1)
# no MAX_PARAMS: capacity study above the d7 deployment budget (prereg)
[ -n "$STOPS" ] && ENVX+=(STOP_AFTER_STEPS=$STOPS)
[ -n "$SAVE_FROM" ] && ENVX+=(SAVE_EPOCHS_FROM=$SAVE_FROM)
[ -n "${RESUME_OUT:-}" ] && ENVX+=(RESUME_WEIGHTS="$LAST_W" INITIAL_EPOCH=$DONE_EP
      RESUME_BEST_JSON="$BEST_J" RESUME_HISTORY="$OUT/seed0/resume_history.json")
{ echo "code $(cat "$HERE/../../CODE_COMMIT.txt" 2>/dev/null | head -1)"
  echo "run $RUN part $PART L=$L width $W params $GOT peak_lr $LRV tf32_train $TF32"
  printf '%s\n' "${ENVX[@]}"; } > "$OUT/env.txt"
status "started L=$L u$W lr=$LRV params=$GOT${RESUME_OUT:+ resumed from epoch $DONE_EP} -> $(basename "$OUT")"

pod () {   # container name, command...
  local name=$1; shift
  podman run --rm --name "$name" --security-opt=label=disable --device nvidia.com/gpu=all \
    -v "$HOME:$HOME" -e HOME="$HOME" -e TF_DETERMINISTIC_OPS=1 -e TF_CUDNN_DETERMINISTIC=1 \
    -e MALLOC_ARENA_MAX=2 "$@"
}
CN="dep_$RUN"
pod "$CN" -w "$SRC" qdec:tf215 env "${ENVX[@]}" bash gh200_run_student_scaling.sh > "$OUT/podman.log" 2>&1 &
PODPID=$!
last=0
while kill -0 $PODPID 2>/dev/null; do
  sleep 60
  "$HOSTPY" "$HERE/classify_lr_scan.py" --kill-check "$OUT" > "$OUT/kill_check.txt" 2>&1
  rc=$?
  ep=$(grep -cE '^[0-9]+/[0-9]+ - [0-9]+s - loss' "$OUT/seed0/run.log" 2>/dev/null)
  if [ "${ep:-0}" -gt "$last" ]; then
    status "epoch $ep $(grep -E '^[0-9]+/[0-9]+ - [0-9]+s - loss' "$OUT/seed0/run.log" | tail -1 | grep -oE '(val_)?(loss|ler): [0-9.e+-]*' | tr '\n' ' ')"
    last=$ep
  fi
  if [ $rc = 3 ]; then
    podman stop -t 30 "$CN" > /dev/null 2>&1
    status "killed: $(cat "$OUT"/KILLED "$OUT"/DIVERGED 2>/dev/null)"
    wait $PODPID
    exit 0
  fi
  free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
  [ "$free_gb" -ge 1 ] || status "WARNING: ${free_gb} GB free on /home"
done
wait $PODPID; rc=$?
[ -f "$OUT/seed0/COMPLETE" ] || { status "FAILED: training exit $rc, no COMPLETE (see $(basename "$OUT")/podman.log)"; exit 1; }
status "trained exit $rc"
if [ "$PART" = A ]; then touch "$OUT/DONE"; status "done"; exit 0; fi

W_LIST=(); for e in $( [ -n "${SMOKE:-}" ] && seq 1 "$EPOCHS" || { echo 20; seq 40 "$EPOCHS"; } ); do
  W_LIST+=("$OUT/seed0/epochs/epoch$(printf '%03d' "$e").weights.h5"); done
read -r -a UL <<< "$UNITS"
pod "depsel_$RUN" -w "$HERE/../selection_2026-10-14" qdec:tf215 python3 score_selection.py \
  --model "$RUN" --d 7 --p 0.004 --units "${UL[@]}" --pool "$POOL" \
  --steps-per-epoch $((FRESH / 10000)) --out-csv "$SEL_CSV" "${W_LIST[@]}" > "$OUT/selection.log" 2>&1
status "selection-block scores exit $?"
"$HOSTPY" "$KD/select_checkpoint.py" --model "$RUN" --csv "$SEL_CSV" \
  --min-epoch "$( [ -n "${SMOKE:-}" ] && echo 1 || echo 40)" --epochs "$EPOCHS" \
  --weights-dir "$OUT/seed0/epochs" --out-json "$OUT/selected.json" >> "$OUT/selection.log" 2>&1 \
  || { status "FAILED: checkpoint selection, see selection.log"; exit 1; }
SELW=$("$HOSTPY" -c "import json,sys; print(json.load(open(sys.argv[1]))['weights'])" "$OUT/selected.json")
pod "depeval_$RUN" -w "$SRC" qdec:tf215 python3 eval_student_on_tail.py --student gru --inputs evts --hidden \
  --units "${UL[@]}" --d 7 --p 0.004 --rounds 7 --pool "$POOL" --weights "$SELW" \
  --eval-start 15200000 --n-test 1800000 --batch-size 10000 --no-tf32 --gpu-mem-mib "$GPU_MEM_MIB" \
  --dump-per-shot "$OUT/eval_selected_per_shot.npz" --out-csv "$BASE/eval_selected.csv" > "$OUT/eval_selected.log" 2>&1
rc=$?
status "eval block ($(basename "$SELW")) exit $rc"
[ $rc = 0 ] && { touch "$OUT/DONE"; status "done"; }
