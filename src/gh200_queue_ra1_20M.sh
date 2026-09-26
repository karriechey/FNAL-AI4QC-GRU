#!/usr/bin/env bash
# Created: 2026-09-26
# Last updated: 2026-09-26
#
# Overnight GPU queue (approved by Karrie 2026-09-26): the u140 GRU with reset_after=True,
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
#
# A failed run is logged and the queue moves on. If a run fails by running out of GPU
# memory, the remaining runs of that distance are skipped (same configuration, same
# memory need) and the skip is logged. The sealed partition is never read.
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
        EARLY_STOP=0 RESET_AFTER=1 GPU_MEM_MIB="$GPU_MEM_MIB" MIN_FREE_MIB="$GPU_MEM_MIB" \
        OUT="$OUT" STAMP="$STAMP" \
    bash gh200_run_student_scaling.sh > "$OUT/podman.log" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 ))
  SEEDDIR="$OUT/seed$SEED"
  if [ $rc -eq 0 ] && [ -f "$SEEDDIR/COMPLETE" ]; then
    log "DONE  d=$D seed=$SEED in ${dt}s"
    printf '%s\t%s\tdone\t%s\t%s\n' "$D" "$SEED" "$dt" "$OUT" >> "$STATUS"
    NEW=$(ls "$SEEDDIR"/per_shot_*.npz | head -1)
    OLD=$(ls "$HOME/rcnn_threshold/results/${OLDRUN[$D]}/seed$SEED"/per_shot_*.npz 2>/dev/null | head -1)
    if [ -n "$OLD" ]; then
      ( cd "$REPO/src" && "$PYH" pair_two_decoders.py --a "$NEW" --b "$OLD" \
          --a-name "ra1_d${D}_s${SEED}" --b-name "ra0_d${D}_s${SEED}" \
          --a-column student_correct --b-column student_correct \
          --out-csv "$SEEDDIR/mcnemar_vs_reset_after_false.csv" ) >> "$SEEDDIR/pair_vs_old.log" 2>&1 \
        && log "PAIR  d=$D seed=$SEED vs ${OLDRUN[$D]}/seed$SEED -> mcnemar_vs_reset_after_false.csv" \
        || log "PAIR  d=$D seed=$SEED FAILED (see pair_vs_old.log)"
    else
      log "PAIR  d=$D seed=$SEED: no old per-shot dump for this seed; not paired"
    fi
  else
    if grep -qE "ResourceExhausted|OOM when allocating|out of memory" "$OUT/podman.log" "$SEEDDIR/run.log" 2>/dev/null; then
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
