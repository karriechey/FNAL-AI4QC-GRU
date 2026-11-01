#!/usr/bin/env bash
# Created: 2026-10-29
# Last modified: 2026-10-29
# One X-basis teacher by name (PLAN.md), resumable: the Z teacher's recipe on X-basis fresh shots,
# epoch snapshots in $SNAP_ROOT, the depth_vs_distance kill rule, then selection block, checkpoint
# selection and eval block, all on the X-basis FORMAL pool. QEC_BASIS=x reaches every container.
#   bash run_x.sh x5_u280 | x7_2xu280 | x9_3xu280
set -uo pipefail
RUN=$1
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../../src" && pwd)"
SEL="$(cd "$HERE/../selection_2026-10-14" && pwd)"
KD="$HERE/../kd_pilot_p004_2026-10"
DEPTH="$HERE/../depth_scaling_d7_2026-10"
DVD="$HERE/../depth_vs_distance/dvd.py"
HOSTPY=${HOSTPY:-$HOME/tfenv/bin/python}
STATUS="$HERE/status.log"
SNAP_ROOT=${SNAP_ROOT:-/tmp/kchey_xbasis/snapshots}
POOLS=${POOLS:-/tmp/kchey_xbasis/pools}
export QEC_BASIS=x

# Z teacher it copies: d, units, peak LR, warmup epochs, epochs, fresh seed base, GPU cap (MiB)
case "$RUN" in
  x5_u280)   D=5; UNITS="280";         LRV=0.003;  WU=0; EPOCHS=80;  FSB=12000000000000; GPU_MEM_MIB=8000; Z=f5w_p004 ;;
  x7_2xu280) D=7; UNITS="280 280";     LRV=0.001;  WU=3; EPOCHS=80;  FSB=13000000000000; GPU_MEM_MIB=8000; Z=f7w_p004 ;;
  x9_3xu280) D=9; UNITS="280 280 280"; LRV=0.0005; WU=3; EPOCHS=240; FSB=14000000000000; GPU_MEM_MIB=10000; Z=f3280e240 ;;
  *) echo "unknown run $RUN"; exit 1 ;;
esac
SEED=0
POOL=$(ls "$POOLS"/data_d${D}_p0.004_r${D}_X_FORMAL_seed*.npz 2>/dev/null | head -1)
[ -s "$POOL" ] || { echo "no X-basis pool for d$D in $POOLS"; exit 1; }
status () { echo "$(date -u +%FT%TZ) $RUN $*" >> "$STATUS"; }
report () { flock "$HERE/.report.lock" "$HOSTPY" "$HERE/report_x.py" > /dev/null 2>&1; }
epochs_done () { "$HOSTPY" -c "import sys; sys.path.insert(0, sys.argv[2]); import dvd; e = dvd.epochs_of(sys.argv[1]); print(max(e) if e else 0)" "$1" "$(dirname "$DVD")"; }
mkdir -p "$HERE/runs" "$SNAP_ROOT"
CN="xb_$RUN"
podman ps --format '{{.Names}}' | grep -qx "$CN" && { echo "$RUN is running"; exit 0; }
EVAL=$(cd "$SRC" && "$HOSTPY" -c "import partitions as P; a, b = P.block('eval'); print(a, b - a)")
read -r EVAL_START N_EVAL <<< "$EVAL"

OUT=$(ls -d "$HERE"/runs/${RUN}_2* 2>/dev/null | tail -1)
DONE_EP=0
if [ -n "$OUT" ]; then
  for m in DONE KILLED DIVERGED; do [ -f "$OUT/$m" ] && { echo "$RUN already $m"; exit 0; }; done
  rm -f "$OUT/FAILED"
  SD="$OUT/seed$SEED"
  if [ ! -f "$SD/COMPLETE" ]; then
    if [ -s "$SD/run.log" ]; then
      n=1; while [ -e "$SD/run_part$n.log" ]; do n=$((n + 1)); done
      prev=$((n - 1))
      { [ "$prev" -ge 1 ] && cmp -s "$SD/run.log" "$SD/run_part$prev.log"; } || cp "$SD/run.log" "$SD/run_part$n.log"
    fi
    DONE_EP=$(epochs_done "$OUT")
    LAST_W=$(printf '%s/epochs/epoch%03d.weights.h5' "$SD" "$DONE_EP")
    if [ "$DONE_EP" -gt 0 ] && [ -s "$LAST_W" ]; then
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
  S=$(date -u +%Y%m%dT%H%M%SZ); OUT="$HERE/runs/${RUN}_$S"; mkdir -p "$OUT/seed$SEED" "$SNAP_ROOT/${RUN}_$S"
  ln -sfn "$SNAP_ROOT/${RUN}_$S" "$OUT/seed$SEED/epochs"
fi
S=${OUT##*_}; SD="$OUT/seed$SEED"

pod () {   # container name, command...
  local name=$1; shift
  podman run --rm --name "$name" --security-opt=label=disable --device nvidia.com/gpu=all \
    -v "$HOME:$HOME" -v /tmp/kchey_xbasis:/tmp/kchey_xbasis -e HOME="$HOME" -e QEC_BASIS=x \
    -e TF_DETERMINISTIC_OPS=1 -e TF_CUDNN_DETERMINISTIC=1 -e MALLOC_ARENA_MAX=2 "$@"
}

ENVX=(D=$D ROUNDS=$D SEEDS=$SEED UNITS="$UNITS" P=0.004 POOL=$POOL QEC_BASIS=x
      NTRAIN=0 FRESH_SHOTS=100000000 FRESH_WORKERS=8 FRESH_SEED_BASE=$FSB
      BATCH=10000 EPOCHS=$EPOCHS LR=$LRV LR_SCHEDULE=cosine LR_END=3e-5 LR_WARMUP_EPOCHS=$WU GLOBAL_CLIPNORM=0.64
      EARLY_STOP=0 RESET_AFTER=0 UNROLL=0 EVAL_NO_TF32=1 TRAIN_NO_TF32=0 SKIP_TAIL_EVAL=1 SAVE_EPOCHS_FROM=1
      GPU_MEM_MIB=$GPU_MEM_MIB MIN_FREE_MIB=$GPU_MEM_MIB MALLOC_ARENA_MAX=2 OUT="$OUT" STAMP="$S")

if [ ! -f "$SD/COMPLETE" ]; then
  RES=()
  if [ "$DONE_EP" -gt 0 ]; then
    RES=(RESUME_WEIGHTS="$(printf '%s/epochs/epoch%03d.weights.h5' "$SD" "$DONE_EP")" INITIAL_EPOCH=$DONE_EP
         RESUME_HISTORY="$SD/resume_history.json")
    BEST_J=$(ls "$SD"/ckpt/*.best.json 2>/dev/null | head -1)
    [ -n "$BEST_J" ] && RES+=(RESUME_BEST_JSON="$BEST_J")
  fi
  { echo "code $(head -1 "$HERE/../../CODE_COMMIT.txt" 2>/dev/null)"
    echo "run $RUN basis x d=$D units $UNITS recipe of $Z"
    printf '%s\n' "${ENVX[@]}" "${RES[@]}"; } > "$OUT/env.txt"
  status "started d=$D units=($UNITS) lr=$LRV wu=$WU epochs=$EPOCHS${RES:+ resumed from epoch $DONE_EP} -> $(basename "$OUT")"
  pod "$CN" -w "$SRC" qdec:tf215 env "${ENVX[@]}" "${RES[@]}" bash gh200_run_student_scaling.sh > "$OUT/podman.log" 2>&1 &
  PODPID=$!
  last=0
  while kill -0 $PODPID 2>/dev/null; do
    sleep 60
    "$HOSTPY" "$DVD" kill-check "$OUT" > "$OUT/kill_check.txt" 2>&1
    rc=$?
    ep=$(epochs_done "$OUT" 2>/dev/null)
    if [ "${ep:-0}" -gt "$last" ]; then
      status "epoch $ep"; last=$ep
      # keep the two newest and every epoch from 40 (selection scoring)
      mapfile -t files < <(ls "$SD"/epochs/epoch*.weights.h5 2>/dev/null | sort)
      for ((i = 0; i < ${#files[@]} - 2; i++)); do
        e=$((10#$(basename "${files[$i]}" .weights.h5 | sed 's/epoch//')))
        [ "$e" -ge 40 ] || rm -f "${files[$i]}"
      done
    fi
    if [ $rc = 3 ]; then
      podman stop -t 30 "$CN" > /dev/null 2>&1; wait $PODPID
      status "killed: $(cat "$OUT"/KILLED "$OUT"/DIVERGED 2>/dev/null)"; report; exit 0
    fi
  done
  wait $PODPID; rc=$?
  [ -f "$SD/COMPLETE" ] || { status "FAILED: training exit $rc (see $(basename "$OUT")/podman.log)"; touch "$OUT/FAILED"; report; exit 1; }
  status "trained"
fi

W_LIST=(); for e in $(seq 40 "$EPOCHS"); do W_LIST+=("$SD/epochs/epoch$(printf '%03d' "$e").weights.h5"); done
for w in "${W_LIST[@]}"; do [ -s "$w" ] || { status "FAILED: missing snapshot $w"; touch "$OUT/FAILED"; report; exit 1; }; done
read -r -a UL <<< "$UNITS"
pod "xbsel_$RUN" -w "$SEL" qdec:tf215 python3 score_selection.py \
  --model "$RUN" --d $D --p 0.004 --units "${UL[@]}" --pool "$POOL" --gpu-mem-mib 10000 \
  --mwpm-cache /tmp/kchey_xbasis/selection_mwpm \
  --steps-per-epoch 10000 --out-csv "$OUT/selection_scores.csv" "${W_LIST[@]}" > "$OUT/selection.log" 2>&1
status "selection-block scores exit $?"
"$HOSTPY" "$KD/select_checkpoint.py" --model "$RUN" --csv "$OUT/selection_scores.csv" --min-epoch 40 --epochs "$EPOCHS" \
  --weights-dir "$SD/epochs" --out-json "$OUT/selected.json" >> "$OUT/selection.log" 2>&1 \
  || { status "FAILED: checkpoint selection, see selection.log"; touch "$OUT/FAILED"; report; exit 1; }
SELW=$("$HOSTPY" -c "import json,sys; print(json.load(open(sys.argv[1]))['weights'])" "$OUT/selected.json")
cp "$SELW" "$OUT/selected.weights.h5"
pod "xbeval_$RUN" -w "$SRC" qdec:tf215 python3 eval_student_on_tail.py --student gru --inputs evts --hidden \
  --units "${UL[@]}" --d $D --p 0.004 --rounds $D --pool "$POOL" --weights "$OUT/selected.weights.h5" \
  --eval-start $EVAL_START --n-test $N_EVAL --batch-size 10000 --no-tf32 --gpu-mem-mib 10000 \
  --dump-per-shot "$OUT/eval_selected_per_shot.npz" --out-csv "$OUT/eval_selected.csv" > "$OUT/eval_selected.log" 2>&1
rc=$?
status "eval block ($(basename "$SELW")) exit $rc"
if [ $rc = 0 ]; then touch "$OUT/DONE"; status "done"; else touch "$OUT/FAILED"; fi
report
