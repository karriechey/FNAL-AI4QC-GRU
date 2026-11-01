#!/usr/bin/env bash
# Created: 2026-11-01 | Last updated: 2026-11-01
# One round-2 student: 1 x GRU <units> distilled with the sweep's best (alpha, T) from choice.json, either
# from the best sweep student (teacher ta) or from f5w_p004 directly (teacher f5w, the control). Recipe,
# shots, selection and eval as in ../kd_sweep_d5_u140_2026-10-31/run_one.sh.
#   bash run_r2.sh ta 48          (SMOKE=1: 2 epochs of 10M shots)
set -uo pipefail
ARM=$1; UNITS=$2
HERE="$(cd "$(dirname "$0")" && pwd)"
PILOT="$(cd "$HERE/../kd_pilot_p004_2026-10" && pwd)"
SRC="$(cd "$HERE/../../src" && pwd)"
HOSTPY=${HOSTPY:-$HOME/tfenv/bin/python}
CHOICE=${CHOICE:-$HERE/choice.json}
get () { "$HOSTPY" -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$CHOICE" "$1"; }
ALPHA=$(get r2_alpha); TEMPERATURE=$(get r2_T)
D=5
source "$PILOT/teachers.sh"                 # f5w_p004, pool, seed base 3e12
if [ "$ARM" = ta ]; then
  source "$HERE/ta_teacher.sh"              # overrides T_NAME, T_UNITS, T_PARAMS, T_COPY, G0
elif [ "$ARM" != f5w ]; then
  echo "arm must be ta or f5w"; exit 1
fi
RUN="r2_${ARM}_u${UNITS}_a${ALPHA}_T${TEMPERATURE}"
LRV=0.003; WU=0; EPOCHS=80
FRESH=100000000; SAVE_FROM=40; BASE=/tmp/kchey_kdsweep/r2_runs; KEEP="$HERE/keep"
if [ -n "${SMOKE:-}" ]; then
  FRESH=10000000; EPOCHS=2; SAVE_FROM=1; BASE=/tmp/kchey_kdsweep/r2_smoke; KEEP="$HERE/smoke_keep"
fi
SEL_CSV="$BASE/selection_scores.csv"
mkdir -p "$BASE" "$KEEP"
[ -f "$G0/summary.json" ] && "$HOSTPY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['passed'] else 1)" "$G0/summary.json" \
  || { echo "teacher check missing or failed ($G0). STOP."; exit 1; }
TF32=$(cat "$G0/tf32_decision.txt")
free_gb=$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 4 ] || { echo "only ${free_gb} GB free on /tmp. STOP."; exit 1; }

S=$(date -u +%Y%m%dT%H%M%SZ); OUT="$BASE/${RUN}_$S"; mkdir -p "$OUT"
KD=(); [ "$ALPHA" != 1.0 ] && KD=(TEACHER_WEIGHTS="$T_COPY" TEACHER_UNITS="$T_UNITS" TEACHER_PARAMS="$T_PARAMS")
ENVX=(D=$D ROUNDS=$D SEEDS=0 UNITS="$UNITS" P=0.004 POOL=$POOL
      NTRAIN=0 FRESH_SHOTS=$FRESH FRESH_WORKERS=8 FRESH_SEED_BASE=$T_FSB
      BATCH=10000 EPOCHS=$EPOCHS LR=$LRV LR_SCHEDULE=cosine LR_END=3e-5 LR_WARMUP_EPOCHS=$WU GLOBAL_CLIPNORM=0.64
      EARLY_STOP=0 RESET_AFTER=0 UNROLL=0 EVAL_NO_TF32=1 TRAIN_NO_TF32=$([ "$TF32" = on ] && echo 0 || echo 1)
      GPU_MEM_MIB=8000 MIN_FREE_MIB=8000 OUT="$OUT" STAMP="$S" MALLOC_ARENA_MAX=2
      SAVE_EPOCHS_FROM=$SAVE_FROM ALPHA=$ALPHA TEMPERATURE=$TEMPERATURE "${KD[@]}")
{ echo "code $(cat "$HERE/../../CODE_COMMIT.txt" 2>/dev/null | head -1)"; echo "run $RUN teacher $T_NAME (sha256 $(sha256sum < "$T_COPY" | cut -c1-16)) tf32_train $TF32"
  printf '%s\n' "${ENVX[@]}"; } > "$OUT/env.txt"
log () { echo "[r2 $(date -u +%FT%TZ)] $*" | tee -a "$BASE/r2.log"; }
log "START $RUN teacher=$T_NAME alpha=$ALPHA T=$TEMPERATURE epochs=$EPOCHS fresh=$FRESH -> $OUT"

pod () {
  local name=$1; shift
  podman run --rm --name "$name" --security-opt=label=disable --device nvidia.com/gpu=all \
    -v "$HOME:$HOME" -v /tmp/kchey_kdsweep:/tmp/kchey_kdsweep -e HOME="$HOME" \
    -e TF_DETERMINISTIC_OPS=1 -e TF_CUDNN_DETERMINISTIC=1 -e MALLOC_ARENA_MAX=2 "$@"
}
pod "r2_$RUN" -w "$SRC" qdec:tf215 env "${ENVX[@]}" bash gh200_run_student_scaling.sh > "$OUT/podman.log" 2>&1
rc=$?; log "trained $RUN exit $rc"
[ -f "$OUT/seed0/COMPLETE" ] || { log "FAIL $RUN: no COMPLETE marker"; exit 1; }

W=(); for e in $(seq "$SAVE_FROM" "$EPOCHS"); do W+=("$OUT/seed0/epochs/epoch$(printf '%03d' "$e").weights.h5"); done
pod "r2sel_$RUN" -w "$HERE/../selection_2026-10-14" qdec:tf215 python3 score_selection.py \
  --model "$RUN" --d "$D" --p 0.004 --units $UNITS --pool "$POOL" \
  --steps-per-epoch $((FRESH / 10000)) --out-csv "$SEL_CSV" "${W[@]}" > "$OUT/selection.log" 2>&1
log "selection-block scores $RUN exit $?"
"$HOSTPY" "$PILOT/select_checkpoint.py" --model "$RUN" --csv "$SEL_CSV" --min-epoch "$SAVE_FROM" \
  --epochs "$EPOCHS" --weights-dir "$OUT/seed0/epochs" --out-json "$OUT/selected.json" >> "$OUT/selection.log" 2>&1 \
  || { log "FAIL $RUN: checkpoint selection, see $OUT/selection.log"; exit 1; }
SELW=$("$HOSTPY" -c "import json,sys; print(json.load(open(sys.argv[1]))['weights'])" "$OUT/selected.json")
pod "r2eval_$RUN" -w "$SRC" qdec:tf215 python3 eval_student_on_tail.py --student gru --inputs evts --hidden \
  --units $UNITS --d "$D" --p 0.004 --rounds "$D" --pool "$POOL" --weights "$SELW" \
  --eval-start 15200000 --n-test 1800000 --batch-size 10000 --no-tf32 --gpu-mem-mib 8000 \
  --dump-per-shot "$OUT/eval_selected_per_shot.npz" --out-csv "$BASE/eval_selected.csv" > "$OUT/eval_selected.log" 2>&1
rc=$?; log "eval block $RUN ($(basename "$SELW")) exit $rc"
[ $rc = 0 ] || exit 1
K="$KEEP/$(basename "$OUT")"; mkdir -p "$K"
cp "$OUT"/{env.txt,selected.json,selection.log,eval_selected.log,eval_selected_per_shot.npz,podman.log} "$SELW" "$K/"
cp "$OUT/seed0/run.log" "$K/run.log" 2>/dev/null
grep "^$RUN," "$SEL_CSV" > "$K/selection_scores.csv"
touch "$OUT/DONE" "$K/DONE"
log "DONE $RUN -> $K"
