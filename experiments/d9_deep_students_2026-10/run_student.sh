#!/usr/bin/env bash
# Created: 2026-10-23
# Last modified: 2026-10-25 (kd9_taper3_t100, kd9_3x134)
# One run of prereg/d9_deep_students_2026-10.md by name. Parameter assertion, resource wait,
# training with the kill rule, then selection-block scoring, selection and the eval block.
#   bash run_student.sh kd9_10x70
#   SMOKE=1 bash run_student.sh kd9_10x70     (2 epochs of 10M shots, for timing)
set -uo pipefail
RUN=$1
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../../src" && pwd)"
# the kd pilot artifacts (teacher copy, teacher check) live in its own deployment on grace1
KD=${KD_DIR:-$HERE/../kd_pilot_p004_2026-10}
[ -d "$KD/teachers" ] || KD=$HOME/FNAL-AI4QC-GRU-kd/experiments/kd_pilot_p004_2026-10
DEPTH="$HERE/../depth_scaling_d7_2026-10"
HOSTPY=${HOSTPY:-$HOME/tfenv/bin/python}
STATUS="$HERE/status.log"
D=9
MAX_PARAMS=303521
# from the smoke test (README, Decisions during the run)
GPU_MEM_MIB=${GPU_MEM_MIB:-8000}
NEED_HOST_GB=${NEED_HOST_GB:-30}
MAX_CONCURRENT=${MAX_CONCURRENT:-3}
MIN_FREE_GB=${MIN_FREE_GB:-3}
# another user's llama-server parks 57 GB on the card at 0% utilization
FOREIGN_MAX_MIB=${FOREIGN_MAX_MIB:-70000}

status () { echo "$(date -u +%FT%TZ) $RUN $*" >> "$STATUS"; }
stop_all () {
  { echo "STOP $(date -u +%FT%TZ) $RUN: $*"; cat "$STATUS" 2>/dev/null; } > "$STATUS.tmp" && mv "$STATUS.tmp" "$STATUS"
  touch "$HERE/STOP"
  exit 1
}

ALPHA=0.0
RUN_EPOCHS=
case "$RUN" in
  kd9_10x70) UNITS=$(printf '70 %.0s' $(seq 10)); WANT=298271 ;;
  kd9_8x70)  UNITS=$(printf '70 %.0s' $(seq 8));  WANT=239051 ;;
  hl9_10x70) UNITS=$(printf '70 %.0s' $(seq 10)); WANT=298271; ALPHA=1.0 ;;
  kd9_taper4) UNITS="160 110 90 70"; WANT=293261 ;;
  kd9_taper3_t100) UNITS="160 120 100"; WANT=283241 ;;
  kd9_3x134) UNITS="134 134 134"; WANT=302841 ;;
  *) echo "unknown run $RUN (README.md lists the runs)"; exit 1 ;;
esac
UNITS=${UNITS% }
[ "$WANT" -le "$MAX_PARAMS" ] || stop_all "prereg count $WANT above the d9 budget $MAX_PARAMS"

GOT=$(cd "$SRC" && CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 "$HOSTPY" -c "
import sys
from StudentModels import build_gru_student
u = [int(x) for x in sys.argv[1].split()]
print(build_gru_student(d=9, rounds=9, inputs='evts', units=tuple(u), hidden=(), p=0.004).count_params())
" "$UNITS" 2>/dev/null | tail -1)
[ "$GOT" = "$WANT" ] || stop_all "parameter assertion failed: units ($UNITS) count_params $GOT, prereg $WANT"

source "$KD/teachers.sh"
[ -f "$G0/summary.json" ] && "$HOSTPY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['passed'] else 1)" "$G0/summary.json" \
  || stop_all "kd pilot teacher check for d9 missing or failed ($G0)"
TF32=$(cat "$G0/tf32_decision.txt")
[ "$TF32" = on ] || [ "$TF32" = off ] || stop_all "no d9 TF32 decision at $G0/tf32_decision.txt"

EPOCHS=${RUN_EPOCHS:-$(cat "$HERE/epochs.txt" 2>/dev/null || echo 240)}
FRESH=100000000; WU=3; SAVE_FROM=40; BASE="$HERE/runs"
if [ -n "${SMOKE:-}" ]; then
  FRESH=10000000; EPOCHS=2; WU=0; SAVE_FROM=1; BASE="$HERE/smoke"
fi
SEL_CSV="$BASE/selection_scores.csv"
mkdir -p "$BASE"

# wait for room: disk, host memory, concurrency, GPU memory, other users on the GPU
waited=""
while true; do
  [ -f "$HERE/STOP" ] && { status "not started: STOP file present"; exit 1; }
  free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
  avail_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  ours=$(podman ps --format '{{.Names}}' | grep -c '^d9_')
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

if [ -n "${RESUME_OUT:-}" ]; then      # set by resume_student.sh after a crash
  OUT="$RESUME_OUT"; S=${OUT##*_}
else
  S=$(date -u +%Y%m%dT%H%M%SZ); OUT="$BASE/${RUN}_$S"; mkdir -p "$OUT"
fi
KDENV=(); [ "$ALPHA" = 0.0 ] && KDENV=(TEACHER_WEIGHTS="$T_COPY" TEACHER_UNITS="$T_UNITS" TEACHER_PARAMS="$T_PARAMS")
ENVX=(D=9 ROUNDS=9 SEEDS=0 UNITS="$UNITS" P=0.004 POOL=$POOL
      NTRAIN=0 FRESH_SHOTS=$FRESH FRESH_WORKERS=8 FRESH_SEED_BASE=$T_FSB
      BATCH=10000 EPOCHS=$EPOCHS LR=5e-4 LR_SCHEDULE=cosine LR_END=3e-5 LR_WARMUP_EPOCHS=$WU GLOBAL_CLIPNORM=0.64
      EARLY_STOP=0 RESET_AFTER=0 UNROLL=0 EVAL_NO_TF32=1 TRAIN_NO_TF32=$([ "$TF32" = on ] && echo 0 || echo 1)
      GPU_MEM_MIB=$GPU_MEM_MIB MIN_FREE_MIB=$GPU_MEM_MIB OUT="$OUT" STAMP="$S" MALLOC_ARENA_MAX=2
      MAX_PARAMS=$MAX_PARAMS SAVE_EPOCHS_FROM=$SAVE_FROM ALPHA=$ALPHA TEMPERATURE=1.0 SKIP_TAIL_EVAL=1 "${KDENV[@]}" "${KDENV[@]}")
[ -n "${RESUME_OUT:-}" ] && ENVX+=(RESUME_WEIGHTS="$RESUME_WEIGHTS" INITIAL_EPOCH=$INITIAL_EPOCH
      RESUME_BEST_JSON="${RESUME_BEST_JSON:-}" RESUME_HISTORY="$RESUME_HISTORY")
{ echo "code $(cat "$HERE/../../CODE_COMMIT.txt" 2>/dev/null | head -1)"
  echo "run $RUN d=9 units ($UNITS) params $GOT alpha $ALPHA epochs $EPOCHS tf32_train $TF32"
  [ "$ALPHA" = 0.0 ] && echo "teacher $T_NAME sha256 $(sha256sum < "$T_COPY" | cut -c1-16)"
  printf '%s\n' "${ENVX[@]}"; } > "$OUT/env.txt"
status "started units ($UNITS) params=$GOT alpha=$ALPHA epochs=$EPOCHS${RESUME_OUT:+ resumed from epoch $INITIAL_EPOCH} -> $(basename "$OUT")"

pod () {   # container name, command...
  local name=$1; shift
  podman run --rm --name "$name" --security-opt=label=disable --device nvidia.com/gpu=all \
    -v "$HOME:$HOME" -e HOME="$HOME" -e TF_DETERMINISTIC_OPS=1 -e TF_CUDNN_DETERMINISTIC=1 \
    -e MALLOC_ARENA_MAX=2 "$@"
}
CN="d9_$RUN"
pod "$CN" -w "$SRC" qdec:tf215 env "${ENVX[@]}" bash gh200_run_student_scaling.sh > "$OUT/podman.log" 2>&1 &
PODPID=$!
last=0; paired=0; disk_checked=$(date +%s)
while kill -0 $PODPID 2>/dev/null; do
  sleep 60
  "$HOSTPY" "$DEPTH/classify_lr_scan.py" --kill-check "$OUT" > "$OUT/kill_check.txt" 2>&1
  rc=$?
  ep=$(grep -cE '^[0-9]+/[0-9]+ - [0-9]+s - loss' "$OUT/seed0/run.log" 2>/dev/null)
  if [ "${ep:-0}" -gt "$last" ]; then
    status "epoch $ep $(grep -E '^[0-9]+/[0-9]+ - [0-9]+s - loss' "$OUT/seed0/run.log" | tail -1 | grep -oE '(val_)?(loss|ler): [0-9.e+-]*' | tr '\n' ' ')"
    last=$ep
  fi
  if [ "${ep:-0}" -gt 0 ] && [ $((ep % 10)) = 0 ] && [ "$ep" != "$paired" ]; then
    paired=$ep
    "$HOSTPY" "$HERE/pair_epoch.py" --run "$RUN" --out "$OUT" --epoch "$ep" >> "$STATUS" 2>&1
  fi
  if [ $rc = 3 ]; then
    podman stop -t 30 "$CN" > /dev/null 2>&1
    status "killed: $(cat "$OUT"/KILLED "$OUT"/DIVERGED 2>/dev/null)"
    wait $PODPID
    exit 0
  fi
  now=$(date +%s)
  if [ $((now - disk_checked)) -ge 43200 ]; then
    status "disk: $(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9) GB free on /home"
    disk_checked=$now
  fi
done
wait $PODPID; rc=$?
[ -f "$OUT/seed0/COMPLETE" ] || { status "FAILED: training exit $rc, no COMPLETE (see $(basename "$OUT")/podman.log)"; exit 1; }
status "trained exit $rc"

read -r -a UL <<< "$UNITS"
W=(); for e in $(seq "$SAVE_FROM" "$EPOCHS"); do W+=("$OUT/seed0/epochs/epoch$(printf '%03d' "$e").weights.h5"); done
pod "d9sel_$RUN" -w "$HERE/../selection_2026-10-14" qdec:tf215 python3 score_selection.py \
  --model "$RUN" --d 9 --p 0.004 --units "${UL[@]}" --pool "$POOL" \
  --steps-per-epoch $((FRESH / 10000)) --out-csv "$SEL_CSV" "${W[@]}" > "$OUT/selection.log" 2>&1
status "selection-block scores exit $?"
"$HOSTPY" "$KD/select_checkpoint.py" --model "$RUN" --csv "$SEL_CSV" --min-epoch "$SAVE_FROM" \
  --epochs "$EPOCHS" --weights-dir "$OUT/seed0/epochs" --out-json "$OUT/selected.json" >> "$OUT/selection.log" 2>&1 \
  || { status "FAILED: checkpoint selection, see selection.log"; exit 1; }
SELW=$("$HOSTPY" -c "import json,sys; print(json.load(open(sys.argv[1]))['weights'])" "$OUT/selected.json")
pod "d9eval_$RUN" -w "$SRC" qdec:tf215 python3 eval_student_on_tail.py --student gru --inputs evts --hidden \
  --units "${UL[@]}" --d 9 --p 0.004 --rounds 9 --pool "$POOL" --weights "$SELW" \
  --eval-start 15200000 --n-test 1800000 --batch-size 10000 --no-tf32 --gpu-mem-mib "$GPU_MEM_MIB" \
  --dump-per-shot "$OUT/eval_selected_per_shot.npz" --out-csv "$BASE/eval_selected.csv" > "$OUT/eval_selected.log" 2>&1
rc=$?
status "eval block ($(basename "$SELW")) exit $rc"
[ $rc = 0 ] || exit 1
touch "$OUT/DONE"; status "done"

# scored: keep the selected epoch and the last, drop the rest (prereg, Resources)
if [ -z "${SMOKE:-}" ]; then
  keep="$(basename "$SELW") epoch$(printf '%03d' "$EPOCHS").weights.h5"
  n=0
  for f in "$OUT"/seed0/epochs/epoch*.weights.h5; do
    case "$keep" in *"$(basename "$f")"*) continue ;; esac
    rm -f "$f"; n=$((n + 1))
  done
  status "pruned $n scored epoch files, kept $keep; $(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9) GB free on /home"
fi
