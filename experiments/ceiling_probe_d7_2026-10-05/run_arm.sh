#!/usr/bin/env bash
# Created: 2026-10-05
# Last modified: 2026-10-06 (arms c1 and d1)
#
# d=7 capacity ceiling probe (diagnostic only, not a deployment candidate; both arms exceed
# the u280 parameter budget on purpose). R01's data (FORMAL prefix 15M + training-only seeds
# 43, 44, 45, 46, 47 = 200M), R01's recipe except the peak LR: cosine from 1e-3 to 3e-5 with a
# 3-epoch warmup (200M at 3e-3 was unstable for u200), Adam global_clipnorm 0.64, batch 10,000,
# 80 epochs, seed 0, reset_after=False, best checkpoint by validation p_L, scored on the
# evaluation block [15.2M, 17M) with MWPM and McNemar (launcher), then on the training slice
# [0, 2M) for the train/eval gap. The sealed block is not read.
#   arm a: one GRU layer of 560 units (1,023,681 parameters)
#   arm b: two stacked GRU layers of 280 units (747,881 parameters)
#   arm c1 (2026-10-06): one u280 layer (276,641), the control for arm a with the same
#           recipe (R01 itself used peak 3e-3)
#   arm d1 (2026-10-06): u280 distilled from arm a's best checkpoint (u560, epoch 38),
#           soft targets only (alpha 0, T 1), teacher logits computed by train_student.py
#           on the training array itself; otherwise the C1 recipe
#
#     bash run_arm.sh a|b        (on grace1, from this folder)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../../src" && pwd)"
arm="$1"
KD=()
case "$arm" in
  a) UNITS="560"; CAP=8000 ;;
  b) UNITS="280 280"; CAP=6000 ;;
  c1) UNITS="280"; CAP=4000 ;;
  d1) UNITS="280"; CAP=8000
      TEACHER=$(ls $HOME/FNAL-AI4QC-GRU-mwpm/experiments/ceiling_probe_d7_2026-10-05/a_u560_*/seed0/ckpt/*.best.keras)
      KD=(ALPHA=0.0 TEMPERATURE=1.0 TEACHER_MODEL="$TEACHER") ;;
  *) echo "arm a, b, c1 or d1"; exit 1 ;;
esac
P7=$HOME/pools_d7_p004_19M
S=$(date -u +%Y%m%dT%H%M%SZ); OUT="$HERE/${arm}_u$(echo $UNITS | tr ' ' 'x')_$S"; mkdir -p "$OUT"
ENVX=(D=7 ROUNDS=7 SEEDS=0 UNITS="$UNITS" P=0.004 POOL=$P7/data_d7_p0.004_r7_FORMAL.npz NTRAIN=15000000
      EXTRA_POOL="$P7/data_d7_p0.004_r7_TRAINONLY_seed43.npz $P7/data_d7_p0.004_r7_TRAINONLY_seed44.npz $P7/data_d7_p0.004_r7_TRAINONLY_seed45.npz $P7/data_d7_p0.004_r7_TRAINONLY_seed46.npz $P7/data_d7_p0.004_r7_TRAINONLY_seed47.npz"
      EXTRA_N="5000000 30000000 50000000 50000000 50000000"
      BATCH=10000 EPOCHS=80 LR=0.001 LR_SCHEDULE=cosine LR_END=3e-5 LR_WARMUP_EPOCHS=3 GLOBAL_CLIPNORM=0.64
      EARLY_STOP=0 RESET_AFTER=0 UNROLL=0 EVAL_NO_TF32=1 GPU_MEM_MIB=$CAP MIN_FREE_MIB=$CAP
      OUT="$OUT" STAMP="$S" MALLOC_ARENA_MAX=2 "${KD[@]}")
{ echo "at_start repo_head $(git -C "$HERE/../.." rev-parse HEAD 2>/dev/null)"; printf '%s\n' "${ENVX[@]}"; } > "$OUT/env.txt"
echo "[probe $(date -u +%FT%TZ)] START arm $arm units=($UNITS) -> $OUT (host available $(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo) GB)" | tee -a "$HERE/probe.log"
podman run --rm --name "probe_$arm" --security-opt=label=disable --device nvidia.com/gpu=all \
  -v "$HOME:$HOME" -e HOME="$HOME" -w "$SRC" qdec:tf215 env "${ENVX[@]}" bash gh200_run_student_scaling.sh \
  > "$OUT/podman.log" 2>&1
echo "[probe $(date -u +%FT%TZ)] trained arm $arm exit $?" | tee -a "$HERE/probe.log"
W=$(ls "$OUT"/seed0/ckpt/*.best.weights.h5)
podman run --rm --name "probe_${arm}_trainslice" --security-opt=label=disable --device nvidia.com/gpu=all \
  -v "$HOME:$HOME" -e HOME="$HOME" -e TF_DETERMINISTIC_OPS=1 -e TF_CUDNN_DETERMINISTIC=1 -e MALLOC_ARENA_MAX=2 \
  -w "$SRC" qdec:tf215 python3 eval_student_on_tail.py --student gru --inputs evts --hidden --units $UNITS \
    --d 7 --p 0.004 --rounds 7 --pool $P7/data_d7_p0.004_r7_FORMAL.npz --weights "$W" \
    --eval-start 0 --n-test 2000000 --batch-size 10000 --no-tf32 --gpu-mem-mib $CAP \
    --out-csv "$HERE/train_slice_scores.csv" > "$OUT/train_slice.log" 2>&1
echo "[probe $(date -u +%FT%TZ)] arm $arm train-slice scored exit $?" | tee -a "$HERE/probe.log"
