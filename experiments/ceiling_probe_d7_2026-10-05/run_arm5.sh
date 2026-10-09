#!/usr/bin/env bash
# Created: 2026-10-05
# Last modified: 2026-10-09 (copy of run_arm4.sh, which is running, with fresh-shot arms: f3160, f3100 and the smoke test fsmoke train on new Stim shots every epoch, FRESH_SHOTS; no training array). Earlier 2026-10-09 (copy of run_arm3.sh, which is running; arm 9s3100: d9 3 x u100 at peak 1e-3, a teacher size that a distilled FPGA student can follow). Earlier 2026-10-08 (copy of run_arm2.sh with the peak LR as a variable; arm 9s3280lr5: d9 3 x u280 at peak 5e-4). Earlier 2026-10-07 (arms 9s16: d9 2 x u160; 9s316: d9 3 x u160; 9s3280: d9 3 x u280). Earlier 2026-10-06 (copy of run_arm.sh with arm 9se, so the copy in use by running jobs is not edited)
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
#   arm s1: two stacked GRU layers of 160 units (254,561 parameters, matched to u280)
#   arm d2: u280 distilled from arm b's best checkpoint (2 x u280, epoch 59), alpha 0, T 1
#   arm s2: two stacked GRU layers of 140 units (197,541 parameters)
#   arm 9s16 (2026-10-07): d=9, two stacked GRU layers of 160 units, the 9-1 / 9-S recipe
#   arms 9s316, 9s3280 (2026-10-07): d=9, three stacked GRU layers of 160 / 280 units, same recipe
#   arm 9s3280lr5 (2026-10-08): 3 x u280 at d9 with peak LR 5e-4 (the 1e-3 run's val p_L drifted
#           up from epoch 22 with rising training loss; stopped at the user's request)
#   arms f3160, f3100 (2026-10-09): 3 x u160 and 3 x u100 at d9, peak 1e-3, 160 epochs of 100M NEW
#           shots each (16B distinct shots; train_student.py --fresh-shots-per-epoch). Same
#           recipe as 9s316 / 9s3100, which train on a fixed 100M and are the controls: 9s
#           scores 0.78x MWPM on its own training shots and 1.62x on the evaluation block.
#   arm fsmoke: 3 x u160, 2 epochs of 5M fresh shots; checks the sampling keeps up with the GPU
#   arm 9s3100 (2026-10-09): 3 x u100 at d9, peak LR 1e-3 (same recipe as 9s316); 175,001 params,
#           a teacher size closer to the FPGA students it will be distilled into
#   arm 9se: 9s with EMA of the weights (decay 0.999), queued only if 9-7 (EMA) is clearly
#           better than 9-1 on the evaluation block
#   arm 9s: d=9, two stacked GRU layers of 280 units (774,761 parameters), the 9-1 recipe:
#           100M (FORMAL 15M + seeds 48, 49), 160 epochs; tests whether depth helps d=9
#
#     bash run_arm.sh a|b        (on grace1, from this folder)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../../src" && pwd)"
arm="$1"
KD=(); D=7; EPOCHS=80; LRV=0.001; FRESH=
case "$arm" in
  a) UNITS="560"; CAP=8000 ;;
  b) UNITS="280 280"; CAP=6000 ;;
  c1) UNITS="280"; CAP=4000 ;;
  d1) UNITS="280"; CAP=8000
      TEACHER=$(ls $HOME/FNAL-AI4QC-GRU-mwpm/experiments/ceiling_probe_d7_2026-10-05/a_u560_*/seed0/ckpt/*.best.keras)
      KD=(ALPHA=0.0 TEMPERATURE=1.0 TEACHER_MODEL="$TEACHER") ;;
  s1) UNITS="160 160"; CAP=6000 ;;
  d2) UNITS="280"; CAP=8000
      TEACHER=$(ls $HOME/FNAL-AI4QC-GRU-probe-b/experiments/ceiling_probe_d7_2026-10-05/b_u280x280_*/seed0/ckpt/*.best.keras)
      KD=(ALPHA=0.0 TEMPERATURE=1.0 TEACHER_MODEL="$TEACHER") ;;
  s2) UNITS="140 140"; CAP=6000 ;;
  9s) UNITS="280 280"; CAP=6000; D=9; EPOCHS=160 ;;
  9se) UNITS="280 280"; CAP=6000; D=9; EPOCHS=160; KD=(EMA_DECAY=0.999) ;;
  9s16) UNITS="160 160"; CAP=6000; D=9; EPOCHS=160 ;;
  9s316) UNITS="160 160 160"; CAP=6000; D=9; EPOCHS=160 ;;
  9s3280) UNITS="280 280 280"; CAP=8000; D=9; EPOCHS=160 ;;
  9s3280lr5) UNITS="280 280 280"; CAP=8000; D=9; EPOCHS=160; LRV=0.0005 ;;
  9s3100) UNITS="100 100 100"; CAP=8000; D=9; EPOCHS=160 ;;
  f3160) UNITS="160 160 160"; CAP=6000; D=9; EPOCHS=160; FRESH=100000000 ;;
  f3100) UNITS="100 100 100"; CAP=6000; D=9; EPOCHS=160; FRESH=100000000 ;;
  fsmoke) UNITS="160 160 160"; CAP=6000; D=9; EPOCHS=2; FRESH=5000000 ;;
  *) echo "arm a, b, c1, d1, s1, d2, s2, 9s, 9se, 9s16, 9s316, 9s3280, 9s3280lr5, 9s3100, f3160, f3100 or fsmoke"; exit 1 ;;
esac
P7=$HOME/pools_d${D}_p004_19M
S=$(date -u +%Y%m%dT%H%M%SZ); OUT="$HERE/${arm}_u$(echo $UNITS | tr ' ' 'x')_$S"; mkdir -p "$OUT"
if [ "$D" = 9 ]; then
  XP="$P7/data_d9_p0.004_r9_TRAINONLY_seed48.npz $P7/data_d9_p0.004_r9_TRAINONLY_seed49.npz"; XN="35000000 50000000"
else
  XP="$P7/data_d7_p0.004_r7_TRAINONLY_seed43.npz $P7/data_d7_p0.004_r7_TRAINONLY_seed44.npz $P7/data_d7_p0.004_r7_TRAINONLY_seed45.npz $P7/data_d7_p0.004_r7_TRAINONLY_seed46.npz $P7/data_d7_p0.004_r7_TRAINONLY_seed47.npz"
  XN="5000000 30000000 50000000 50000000 50000000"
fi
# Fresh-shot arms: no training prefix and no extension pools; FRESH new shots every epoch.
if [ -n "$FRESH" ]; then TRAINDATA=(NTRAIN=0 FRESH_SHOTS=$FRESH FRESH_WORKERS=8)
else TRAINDATA=(NTRAIN=15000000 EXTRA_POOL="$XP" EXTRA_N="$XN"); fi
ENVX=(D=$D ROUNDS=$D SEEDS=0 UNITS="$UNITS" P=0.004 POOL=$P7/data_d${D}_p0.004_r${D}_FORMAL.npz
      "${TRAINDATA[@]}"
      BATCH=10000 EPOCHS=$EPOCHS LR=$LRV LR_SCHEDULE=cosine LR_END=3e-5 LR_WARMUP_EPOCHS=3 GLOBAL_CLIPNORM=0.64
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
    --d $D --p 0.004 --rounds $D --pool $P7/data_d${D}_p0.004_r${D}_FORMAL.npz --weights "$W" \
    --eval-start 0 --n-test 2000000 --batch-size 10000 --no-tf32 --gpu-mem-mib $CAP \
    --out-csv "$HERE/train_slice_scores.csv" > "$OUT/train_slice.log" 2>&1
echo "[probe $(date -u +%FT%TZ)] arm $arm train-slice scored exit $?" | tee -a "$HERE/probe.log"
