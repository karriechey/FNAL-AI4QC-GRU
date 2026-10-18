#!/usr/bin/env bash
# Created: 2026-10-18
# Last modified: 2026-10-18
#   bash run_kd.sh kd5_u140          (SMOKE=1: 2 epochs of 10M shots, for timing)
set -uo pipefail
RUN=$1
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../../src" && pwd)"
HOSTPY=${HOSTPY:-$HOME/tfenv/bin/python}
ALPHA=0.0
case "$RUN" in
  kd5_u140|hl5_u140) D=5; UNITS="140" ;;
  kd5_u70|hl5_u70) D=5; UNITS="70" ;;
  kd7_2x140|hl7_2x140) D=7; UNITS="140 140" ;;
  kd7_2x70|hl7_2x70) D=7; UNITS="70 70" ;;
  kd7_1x113) D=7; UNITS="113" ;;
  kd9_3x140) D=9; UNITS="140 140 140" ;;
  kd9_3x70|hl9_3x70) D=9; UNITS="70 70 70" ;;
  kd9_1x140) D=9; UNITS="140" ;;
  *) echo "unknown run $RUN (README.md lists the matrix)"; exit 1 ;;
esac
[ "${RUN:0:2}" = hl ] && ALPHA=1.0
source "$HERE/teachers.sh"
case "$D" in             # teacher recipe
  5) LRV=0.003; WU=0; EPOCHS=80 ;;
  7) LRV=0.001; WU=3; EPOCHS=80 ;;
  9) LRV=0.0005; WU=3; EPOCHS=160 ;;
esac
FRESH=100000000; SAVE_FROM=40; BASE="$HERE/runs"
if [ -n "${SMOKE:-}" ]; then
  FRESH=10000000; EPOCHS=2; WU=0; SAVE_FROM=1; BASE="$HERE/smoke"
fi
SEL_CSV="$BASE/selection_scores.csv"

# pre-registered: hl9_3x70 only if kd9_3x70 lands in [1.0x, 2.3x] MWPM on the selection block
if [ "$RUN" = hl9_3x70 ] && [ -z "${SMOKE:-}" ]; then
  sel=$(ls "$BASE"/kd9_3x70_*/selected.json 2>/dev/null | head -1)
  [ -n "$sel" ] || { echo "hl9_3x70: kd9_3x70 has no selected checkpoint yet. STOP."; exit 1; }
  "$HOSTPY" -c "import json,sys; r=json.load(open(sys.argv[1]))['sel_ratio']; sys.exit(0 if 1.0 <= r <= 2.3 else 1)" "$sel" \
    || { echo "hl9_3x70: kd9_3x70 selection ratio outside [1.0, 2.3], not run (pre-registered). STOP."; exit 1; }
fi
[ -f "$G0/summary.json" ] && "$HOSTPY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['passed'] else 1)" "$G0/summary.json" \
  || { echo "teacher check for d$D missing or failed ($G0). STOP."; exit 1; }
TF32=$(cat "$G0/tf32_decision.txt")
free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge "${MIN_FREE_GB:-4}" ] || { echo "only ${free_gb} GB free on /home. STOP."; exit 1; }

S=$(date -u +%Y%m%dT%H%M%SZ); OUT="$BASE/${RUN}_$S"; mkdir -p "$OUT"
KD=(); [ "$ALPHA" = 0.0 ] && KD=(TEACHER_WEIGHTS="$T_COPY" TEACHER_UNITS="$T_UNITS" TEACHER_PARAMS="$T_PARAMS")
ENVX=(D=$D ROUNDS=$D SEEDS=0 UNITS="$UNITS" P=0.004 POOL=$POOL
      NTRAIN=0 FRESH_SHOTS=$FRESH FRESH_WORKERS=8 FRESH_SEED_BASE=$T_FSB
      BATCH=10000 EPOCHS=$EPOCHS LR=$LRV LR_SCHEDULE=cosine LR_END=3e-5 LR_WARMUP_EPOCHS=$WU GLOBAL_CLIPNORM=0.64
      EARLY_STOP=0 RESET_AFTER=0 UNROLL=0 EVAL_NO_TF32=1 TRAIN_NO_TF32=$([ "$TF32" = on ] && echo 0 || echo 1)
      GPU_MEM_MIB=8000 MIN_FREE_MIB=8000 OUT="$OUT" STAMP="$S" MALLOC_ARENA_MAX=2
      SAVE_EPOCHS_FROM=$SAVE_FROM ALPHA=$ALPHA TEMPERATURE=1.0 "${KD[@]}")
{ echo "code $(cat "$HERE/../../CODE_COMMIT.txt" 2>/dev/null | head -1)"; echo "run $RUN teacher $T_NAME (sha256 $(sha256sum < "$T_COPY" | cut -c1-16)) tf32_train $TF32"
  printf '%s\n' "${ENVX[@]}"; } > "$OUT/env.txt"
log () { echo "[kd $(date -u +%FT%TZ)] $*" | tee -a "$BASE/pilot.log"; }
log "START $RUN d=$D units=($UNITS) alpha=$ALPHA epochs=$EPOCHS fresh=$FRESH -> $OUT"

pod () {   # container name, command...
  local name=$1; shift
  podman run --rm --name "$name" --security-opt=label=disable --device nvidia.com/gpu=all \
    -v "$HOME:$HOME" -e HOME="$HOME" -e TF_DETERMINISTIC_OPS=1 -e TF_CUDNN_DETERMINISTIC=1 \
    -e MALLOC_ARENA_MAX=2 "$@"
}
pod "kd_$RUN" -w "$SRC" qdec:tf215 env "${ENVX[@]}" bash gh200_run_student_scaling.sh > "$OUT/podman.log" 2>&1
rc=$?; log "trained $RUN exit $rc"
[ -f "$OUT/seed0/COMPLETE" ] || { log "FAIL $RUN: no COMPLETE marker"; exit 1; }

read -r -a UL <<< "$UNITS"
W=(); for e in $(seq "$SAVE_FROM" "$EPOCHS"); do W+=("$OUT/seed0/epochs/epoch$(printf '%03d' "$e").weights.h5"); done
pod "kdsel_$RUN" -w "$HERE/../selection_2026-10-14" qdec:tf215 python3 score_selection.py \
  --model "$RUN" --d "$D" --p 0.004 --units "${UL[@]}" --pool "$POOL" \
  --steps-per-epoch $((FRESH / 10000)) --out-csv "$SEL_CSV" "${W[@]}" > "$OUT/selection.log" 2>&1
log "selection-block scores $RUN exit $?"
"$HOSTPY" "$HERE/select_checkpoint.py" --model "$RUN" --csv "$SEL_CSV" --min-epoch "$SAVE_FROM" \
  --epochs "$EPOCHS" --weights-dir "$OUT/seed0/epochs" --out-json "$OUT/selected.json" >> "$OUT/selection.log" 2>&1 \
  || { log "FAIL $RUN: checkpoint selection, see $OUT/selection.log"; exit 1; }
SELW=$("$HOSTPY" -c "import json,sys; print(json.load(open(sys.argv[1]))['weights'])" "$OUT/selected.json")
pod "kdeval_$RUN" -w "$SRC" qdec:tf215 python3 eval_student_on_tail.py --student gru --inputs evts --hidden \
  --units "${UL[@]}" --d "$D" --p 0.004 --rounds "$D" --pool "$POOL" --weights "$SELW" \
  --eval-start 15200000 --n-test 1800000 --batch-size 10000 --no-tf32 --gpu-mem-mib 8000 \
  --dump-per-shot "$OUT/eval_selected_per_shot.npz" --out-csv "$BASE/eval_selected.csv" > "$OUT/eval_selected.log" 2>&1
rc=$?; log "eval block $RUN ($(basename "$SELW")) exit $rc"
[ $rc = 0 ] && touch "$OUT/DONE"
