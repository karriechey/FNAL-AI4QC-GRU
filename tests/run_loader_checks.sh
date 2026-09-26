#!/usr/bin/env bash
# Created: 2026-09-26
# Last updated: 2026-09-26 (RA=0 runs (b)/(c) with reset_after=False; SKIP_A=1 skips (a); UR=1 adds --unroll and step (d))
#
# Runs the three host-memory pipeline checks in the qdec:tf215 image on grace1, with the
# launcher's determinism variables and the 3,500 MiB GPU cap. Writes everything to $OUT.
#   (a) coverage: every training index exactly once per epoch (tests/loader_checks.py)
#   (b) determinism: two identical short d=5 runs (200k shots, 3 epochs, seed 0) must
#       give identical per-epoch losses and identical best weights
#   (c) peak GPU memory: one epoch of the real d=5 and d=7 20M-augmented training set
#       (FORMAL [0,15M) + TRAINONLY seed 43 5M), batch 10,000, reset_after=True
set -uo pipefail
REPO="$HOME/FNAL-AI4QC-GRU"
OUT="${OUT:?set OUT}"
mkdir -p "$OUT"
run () {    # run <name> <command...>   (inside the image, GPU visible)
  local name="$1"; shift
  podman run --rm --security-opt=label=disable --device nvidia.com/gpu=all \
    -v "$HOME:$HOME" -e HOME="$HOME" -e TF_DETERMINISTIC_OPS=1 -e TF_CUDNN_DETERMINISTIC=1 \
    -e TF_CPP_MIN_LOG_LEVEL=2 -w "$REPO" qdec:tf215 "$@" > "$OUT/$name.log" 2>&1
  echo "[checks] $name rc=$?" | tee -a "$OUT/summary.log"
}
common () {  # common <d> <n_train> <epochs> <tag> [extra args...]
  local d="$1" ntr="$2" ep="$3" tag="$4"; shift 4
  local raflag=""; [ "${RA:-1}" = "1" ] && raflag="--reset-after"
  [ "${UR:-0}" = "1" ] && raflag="$raflag --unroll"
  echo --student gru --inputs evts --hidden --units 140 $raflag \
       --d "$d" --p 0.004 --rounds "$d" \
       --pool "$HOME/pools_d${d}_p004_19M/data_d${d}_p0.004_r${d}_FORMAL.npz" \
       --n-train "$ntr" --val-start 15000000 --val-n 200000 \
       --eval-start 15200000 --n-test 1800000 --alpha 1.0 --temperature 1.0 \
       --batch-size 10000 --lr 0.003 --epochs "$ep" --no-early-stopping \
       --seed 0 --require-determinism --gpu-mem-mib 3500 \
       --ckpt-dir "$OUT/$tag/ckpt" --out-dir "$OUT/$tag/runs" --tag "$tag" --run-tag "$tag" "$@"
}

[ "${SKIP_A:-0}" = "1" ] || run a_coverage python3 tests/loader_checks.py coverage
for rep in 1 2; do
  run "b_det_run$rep" python3 tests/loader_checks.py train --out "$OUT/b_det_run$rep.json" \
      -- $(common 5 200000 3 "b_det_run$rep")
done
for d in 5 7; do
  run "c_peak_d$d" python3 tests/loader_checks.py train --out "$OUT/c_peak_d$d.json" \
      -- $(common "$d" 15000000 1 "c_peak_d$d" \
           --extra-train-pool "$HOME/pools_d${d}_p004_19M/data_d${d}_p0.004_r${d}_TRAINONLY_seed43.npz" \
           --extra-train-n 5000000)
done
# (d) GPU vs CPU logits of each (c) checkpoint on 10,000 val shots (same model, same weights).
if [ "${UR:-0}" = "1" ]; then
  flags="--unroll"; [ "${RA:-1}" = "1" ] && flags="--reset-after $flags"
  for d in 5 7; do
    W=$(ls "$OUT/c_peak_d$d/ckpt/"*.best.weights.h5 | head -1)
    run "d_logits_gpu_d$d" python3 tests/gpu_cpu_logits.py --weights "$W" --d $d $flags --out "$OUT/d_gpu_d$d.npz"
    podman run --rm --security-opt=label=disable -v "$HOME:$HOME" -e HOME="$HOME" -e TF_CPP_MIN_LOG_LEVEL=2 \
      -w "$REPO" qdec:tf215 python3 tests/gpu_cpu_logits.py --weights "$W" --d $d $flags \
      --out "$OUT/d_cpu_d$d.npz" > "$OUT/d_logits_cpu_d$d.log" 2>&1
    echo "[checks] d_logits_cpu_d$d rc=$?" | tee -a "$OUT/summary.log"
    podman run --rm --security-opt=label=disable -v "$HOME:$HOME" -e HOME="$HOME" -w "$REPO" qdec:tf215 \
      python3 tests/gpu_cpu_logits.py --compare "$OUT/d_gpu_d$d.npz" "$OUT/d_cpu_d$d.npz" 2>/dev/null \
      | grep "^\[gpu-cpu\]" | tee -a "$OUT/summary.log"
  done
fi
echo "[checks] done" | tee -a "$OUT/summary.log"
