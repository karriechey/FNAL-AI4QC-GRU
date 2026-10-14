#!/usr/bin/env bash
# Created: 2026-10-13
# Last modified: 2026-10-13
#
# Copy of run_arm8.sh (which is running f3280s1 and must not be edited) for the per-distance
# winners trained on fresh shots. The winning model and learning rate at each distance, with
# 100M new Stim shots every epoch in place of a fixed training set (at d9 fresh shots turned the
# overfitting stacked GRUs into the best models: 3 x u280 1.52x -> 0.665x MWPM):
#   arm f5w: d5, one GRU layer of 280, the R08 recipe (0.761x on fixed 100M): peak 3e-3, cosine
#           to 3e-5 over 80 epochs, no warmup, clipnorm 0.64
#   arm f7w: d7, two stacked GRU layers of 280, the arm b recipe (0.717x on fixed 200M): peak 1e-3,
#           cosine to 3e-5 over 80 epochs, 3 warmup epochs, clipnorm 0.64
# The d9 winner at p=0.004 is f3280e240 (run_arm8.sh). P sets the physical error rate (default
# 0.004); WU the warmup epochs (default 3). Everything after the arm table is run_arm8.sh's
# launcher, including RESUME_FROM.
#
#     bash run_arm10.sh <arm>        (on grace1, from this folder)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../../src" && pwd)"
arm="$1"
KD=(); D=7; EPOCHS=80; LRV=0.001; FRESH=; SEED=0; FSB=; WU=3; P=0.004
case "$arm" in
  f5w) UNITS="280"; CAP=8000; D=5; EPOCHS=80; FRESH=100000000; LRV=0.003; WU=0 ;;
  f7w) UNITS="280 280"; CAP=8000; D=7; EPOCHS=80; FRESH=100000000; LRV=0.001 ;;
  *) echo "arm f5w or f7w"; exit 1 ;;
esac
PT=$(printf '%03d' "$(python3 -c "print(round($P*1000))")")      # 0.004 -> 004
P7=$HOME/pools_d${D}_p${PT}_19M
S=$(date -u +%Y%m%dT%H%M%SZ); OUT="$HERE/${arm}_p${PT}_u$(echo $UNITS | tr ' ' 'x')_$S"; mkdir -p "$OUT"
RESUME=()
if [ -n "${RESUME_FROM:-}" ]; then
  mapfile -t RESUME < <(python3 "$HERE/prepare_resume.py" "$RESUME_FROM" "$OUT" "$SEED") || exit 1
  [ "${#RESUME[@]}" = 4 ] || { echo "prepare_resume.py failed for $RESUME_FROM"; exit 1; }
fi
if [ "$D" = 9 ]; then
  XP="$P7/data_d9_p0.004_r9_TRAINONLY_seed48.npz $P7/data_d9_p0.004_r9_TRAINONLY_seed49.npz"; XN="35000000 50000000"
else
  XP="$P7/data_d7_p0.004_r7_TRAINONLY_seed43.npz $P7/data_d7_p0.004_r7_TRAINONLY_seed44.npz $P7/data_d7_p0.004_r7_TRAINONLY_seed45.npz $P7/data_d7_p0.004_r7_TRAINONLY_seed46.npz $P7/data_d7_p0.004_r7_TRAINONLY_seed47.npz"
  XN="5000000 30000000 50000000 50000000 50000000"
fi
# Fresh-shot arms: no training prefix and no extension pools; FRESH new shots every epoch.
if [ -n "$FRESH" ]; then TRAINDATA=(NTRAIN=0 FRESH_SHOTS=$FRESH FRESH_WORKERS=8 FRESH_SEED_BASE=$FSB)
else TRAINDATA=(NTRAIN=15000000 EXTRA_POOL="$XP" EXTRA_N="$XN"); fi
ENVX=(D=$D ROUNDS=$D SEEDS=$SEED UNITS="$UNITS" P=$P POOL=$P7/data_d${D}_p${P}_r${D}_FORMAL.npz
      "${TRAINDATA[@]}"
      BATCH=10000 EPOCHS=$EPOCHS LR=$LRV LR_SCHEDULE=cosine LR_END=3e-5 LR_WARMUP_EPOCHS=$WU GLOBAL_CLIPNORM=0.64
      EARLY_STOP=0 RESET_AFTER=0 UNROLL=0 EVAL_NO_TF32=1 GPU_MEM_MIB=$CAP MIN_FREE_MIB=$CAP
      OUT="$OUT" STAMP="$S" MALLOC_ARENA_MAX=2 "${KD[@]}" "${RESUME[@]}")
{ echo "at_start repo_head $(git -C "$HERE/../.." rev-parse HEAD 2>/dev/null)"; printf '%s\n' "${ENVX[@]}"; } > "$OUT/env.txt"
echo "[probe $(date -u +%FT%TZ)] START arm $arm units=($UNITS)${RESUME_FROM:+ resumed from $RESUME_FROM after epoch ${RESUME[1]#INITIAL_EPOCH=}} -> $OUT (host available $(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo) GB)" | tee -a "$HERE/probe.log"
podman run --rm --name "probe_${arm}_p${PT}" --security-opt=label=disable --device nvidia.com/gpu=all \
  -v "$HOME:$HOME" -e HOME="$HOME" -w "$SRC" qdec:tf215 env "${ENVX[@]}" bash gh200_run_student_scaling.sh \
  > "$OUT/podman.log" 2>&1
echo "[probe $(date -u +%FT%TZ)] trained arm $arm exit $?" | tee -a "$HERE/probe.log"
W=$(ls "$OUT"/seed$SEED/ckpt/*.best.weights.h5)
podman run --rm --name "probe_${arm}_p${PT}_trainslice" --security-opt=label=disable --device nvidia.com/gpu=all \
  -v "$HOME:$HOME" -e HOME="$HOME" -e TF_DETERMINISTIC_OPS=1 -e TF_CUDNN_DETERMINISTIC=1 -e MALLOC_ARENA_MAX=2 \
  -w "$SRC" qdec:tf215 python3 eval_student_on_tail.py --student gru --inputs evts --hidden --units $UNITS \
    --d $D --p $P --rounds $D --pool $P7/data_d${D}_p${P}_r${D}_FORMAL.npz --weights "$W" \
    --eval-start 0 --n-test 2000000 --batch-size 10000 --no-tf32 --gpu-mem-mib $CAP \
    --out-csv "$HERE/train_slice_scores.csv" > "$OUT/train_slice.log" 2>&1
echo "[probe $(date -u +%FT%TZ)] arm $arm train-slice scored exit $?" | tee -a "$HERE/probe.log"
