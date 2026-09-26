#!/usr/bin/env bash
# Created: 2026-09-26
# Last updated: 2026-09-26 (also treat 'Dst tensor is not initialized' as out of GPU memory; UNROLL=1; TF32 on for training, off for evaluation; old checkpoints re-scored)
#
# Overnight GPU queue (approved 2026-09-26): the u140 GRU with reset_after=True,
# trained exactly like the old d=5 / d=7 "20M augmented" runs except for reset_after.
#
#   order (sequential)  d5 s0 -> d7 s0 -> d5 s1 -> d7 s1 -> d5 s2
#   data                FORMAL train [0, 15M) + TRAINONLY seed-43 [0, 5M)
#   recipe              batch 10,000, 200 epochs, constant LR 3e-3, early stopping off,
#                       determinism variables on (set by the launcher)
#   selection           lowest val LER on the val block, tiebreak val loss
#   evaluation          eval block only; MWPM recomputed on the same shots (launcher);
#                       then McNemar vs the old reset_after=False checkpoint of the same
#                       d and seed (pair_two_decoders.py), b / c / n_discordant
#   GPU                 3.5 GB cap per run (sglang holds ~91.5 GB)
#   GRU layer           reset_after=True, unroll=True: the generic kernel, as in every old
#                       run. TF 2.15's cuDNN GRU call fails under the image's cuDNN 9.
#   TF32                on for training (TensorFlow default, as in every old run); OFF for every
#                       evaluation: the new best checkpoint (launcher EVAL_NO_TF32=1) and the
#                       old reset_after=False checkpoint of the same d and seed, which is
#                       re-scored here with --no-tf32 on the same eval block before pairing.
#
# A failed run is logged and the queue moves on. If a run fails by running out of GPU
# memory, the remaining runs of that distance are skipped (same configuration, same
# memory need) and the skip is logged. The sealed partition is never read.
#
# C2b (2026-09-26): when d5 s0 completes, its best reset_after=True checkpoint is
# converted with hls4ml in the image (65,536-entry ap_fixed<32,16> tables) and compared
# with Keras on 500 eval-block inputs (tests/c2_hls4ml_gru_check.py, CPU only).
#
# Run detached on grace1 from the repo's src/ folder:
#     setsid nohup bash gh200_queue_ra1_20M.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail

REPO="$HOME/FNAL-AI4QC-GRU"
QROOT="$REPO/experiments/queue_ra1_20M"
QLOG="$QROOT/queue.log"
STATUS="$QROOT/status.tsv"          # one line per job, read by the morning summary
IMG="${IMG:-qdec:tf215}"
GPU_MEM_MIB=3500
JOBS=("5 0" "7 0" "5 1" "7 1" "5 2")
PYH="$HOME/tfenv/bin/python"
mkdir -p "$QROOT"
[ -f "$STATUS" ] || printf 'd\tseed\tstatus\tseconds\tout\n' > "$STATUS"

# Old reset_after=False runs of record with the same data and recipe.
declare -A OLDRUN=( [5]=d5_u140_ntr20M_aug_20260821T154820Z [7]=d7_u140_ntr20M_aug_20260821T193446Z )
declare -A SKIP=()

log () { echo "[queue $(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$QLOG"; }

log "START queue pid=$$ jobs=${#JOBS[@]} image=$IMG gpu_cap=${GPU_MEM_MIB}MiB repo=$(git -C "$REPO" rev-parse --short HEAD)"
for job in "${JOBS[@]}"; do
  read -r D SEED <<< "$job"
  if [ -n "${SKIP[$D]:-}" ]; then
    log "SKIP  d=$D seed=$SEED: ${SKIP[$D]}"
    printf '%s\t%s\tskipped\t0\t-\n' "$D" "$SEED" >> "$STATUS"
    continue
  fi
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  OUT="$QROOT/d${D}_s${SEED}_${STAMP}"
  EXTRA="$HOME/pools_d${D}_p004_19M/data_d${D}_p0.004_r${D}_TRAINONLY_seed43.npz"
  mkdir -p "$OUT"
  log "RUN   d=$D seed=$SEED -> $OUT"
  t0=$(date +%s)
  podman run --rm --security-opt=label=disable --device nvidia.com/gpu=all \
    -v "$HOME:$HOME" -e HOME="$HOME" -w "$REPO/src" "$IMG" \
    env D="$D" ROUNDS="$D" SEEDS="$SEED" UNITS=140 NTRAIN=15000000 \
        EXTRA_POOL="$EXTRA" EXTRA_N=5000000 BATCH=10000 EPOCHS=200 LR=0.003 \
        EARLY_STOP=0 RESET_AFTER=1 UNROLL=1 EVAL_NO_TF32=1 GPU_MEM_MIB="$GPU_MEM_MIB" MIN_FREE_MIB="$GPU_MEM_MIB" \
        OUT="$OUT" STAMP="$STAMP" \
    bash gh200_run_student_scaling.sh > "$OUT/podman.log" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 ))
  SEEDDIR="$OUT/seed$SEED"
  if [ $rc -eq 0 ] && [ -f "$SEEDDIR/COMPLETE" ]; then
    log "DONE  d=$D seed=$SEED in ${dt}s"
    printf '%s\t%s\tdone\t%s\t%s\n' "$D" "$SEED" "$dt" "$OUT" >> "$STATUS"
    NEW=$(ls "$SEEDDIR"/per_shot_*.npz | head -1)
    # Re-score the old reset_after=False checkpoint with TF32 off, same eval block.
    OLDCK="$HOME/rcnn_threshold/results/${OLDRUN[$D]}/seed$SEED/ckpt/gru_d${D}_p004_u140_ext5000000_hard_seed${SEED}_ntr15000000_b10000.best.weights.h5"
    RES="$QROOT/old_rescored_tf32off/d${D}_s${SEED}"
    OLD=""
    if [ -f "$OLDCK" ]; then
      mkdir -p "$RES"
      eval "$("$PYH" "$REPO/src/partitions.py" --shell --d "$D")"
      podman run --rm --security-opt=label=disable --device nvidia.com/gpu=all \
        -v "$HOME:$HOME" -e HOME="$HOME" -w "$REPO/src" "$IMG" \
        python3 eval_student_on_tail.py --student gru --inputs evts --hidden --units 140 \
          --no-tf32 --gpu-mem-mib "$GPU_MEM_MIB" --weights "$OLDCK" \
          --d "$D" --p 0.004 --rounds "$D" --pool "$POOL" \
          --eval-start "$EVAL_START" --n-test "$EVAL_N" --batch-size 10000 \
          --dump-per-shot "$RES/per_shot_old_ra0.npz" --out-csv "$RES/eval_old_ra0.csv" \
          > "$RES/rescore.log" 2>&1 \
        && OLD="$RES/per_shot_old_ra0.npz" && log "RESCORE old d=$D seed=$SEED (TF32 off) -> $RES" \
        || log "RESCORE old d=$D seed=$SEED FAILED (see $RES/rescore.log)"
    fi
    if [ -n "$OLD" ]; then
      ( cd "$REPO/src" && "$PYH" pair_two_decoders.py --a "$NEW" --b "$OLD" \
          --a-name "ra1_d${D}_s${SEED}" --b-name "ra0_d${D}_s${SEED}" \
          --a-column student_correct --b-column student_correct \
          --out-csv "$SEEDDIR/mcnemar_vs_reset_after_false.csv" ) >> "$SEEDDIR/pair_vs_old.log" 2>&1 \
        && log "PAIR  d=$D seed=$SEED vs ${OLDRUN[$D]}/seed$SEED -> mcnemar_vs_reset_after_false.csv" \
        || log "PAIR  d=$D seed=$SEED FAILED (see pair_vs_old.log)"
    else
      log "PAIR  d=$D seed=$SEED: no old checkpoint re-scored for this seed; not paired"
    fi
    if [ "$D" = 5 ] && [ "$SEED" = 0 ]; then
      C2B="$REPO/experiments/gru_check/c2b_$(date -u +%Y%m%dT%H%M%SZ)"
      mkdir -p "$C2B"
      BEST=$(ls "$SEEDDIR"/ckpt/*.best.weights.h5 | head -1)
      podman run --rm --security-opt=label=disable -v "$HOME:$HOME" -e HOME="$HOME" \
        -e TF_CPP_MIN_LOG_LEVEL=3 -w "$REPO" "$IMG" \
        python3 tests/c2_hls4ml_gru_check.py --out "$C2B" --trained-weights "$BEST" \
          --trained-units 140 --trained-reset-after --trained-unroll --no-tf32 --table-size 65536 \
          --table-precision "ap_fixed<32,16>" > "$C2B/run.log" 2>&1 \
        && log "C2B   done on $BEST -> $C2B" || log "C2B   FAILED (see $C2B/run.log)"
    fi
  else
    if grep -qE "ResourceExhausted|OOM when allocating|out of memory|Dst tensor is not initialized" "$OUT/podman.log" "$SEEDDIR/run.log" 2>/dev/null; then
      SKIP[$D]="d=$D seed=$SEED ran out of GPU memory under the ${GPU_MEM_MIB} MiB cap"
      log "FAIL  d=$D seed=$SEED rc=$rc after ${dt}s: out of GPU memory; skipping the remaining d=$D runs"
      printf '%s\t%s\tfailed_oom\t%s\t%s\n' "$D" "$SEED" "$dt" "$OUT" >> "$STATUS"
    else
      log "FAIL  d=$D seed=$SEED rc=$rc after ${dt}s (COMPLETE absent); continuing"
      printf '%s\t%s\tfailed\t%s\t%s\n' "$D" "$SEED" "$dt" "$OUT" >> "$STATUS"
    fi
  fi
done
log "END   queue"
