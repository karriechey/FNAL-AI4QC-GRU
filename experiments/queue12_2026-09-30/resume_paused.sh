#!/usr/bin/env bash
# Created: 2026-10-02
# Last updated: 2026-10-02
#
# Resume the three runs paused on 2026-10-01 to stop swap growth (R04 after epoch 31, R08
# after 36, R12 after 5) from their last completed epoch to 80, with every original setting
# plus MALLOC_ARENA_MAX=2 (memory levels off; losses identical, tests/memory_growth_test.sh).
#
# A stopped run has its .lastepoch.weights.h5 (model + optimizer state after the last
# completed epoch) and .best.* checkpoint, but no history file (that is written at the end).
# For each run this script:
#   1. rebuilds the history of epochs 1..N from the per-epoch lines of run.log (values as
#      printed, 4 significant decimals; it only fills the history file, training does not
#      read it);
#   2. makes a new folder <run>_resume_<stamp>: copies .best.* into seed0/ckpt/ and the
#      lastepoch weights + rebuilt history into resume_from/ (the paused folder is not
#      changed);
#   3. launches the same launcher call as the original (command.txt) with RESUME_WEIGHTS,
#      INITIAL_EPOCH=N, RESUME_BEST_JSON, RESUME_HISTORY. train_student.py stops if the
#      restored optimizer update count is not N x steps per epoch.
# Jobs start one at a time (next one after the previous reaches its first resumed epoch).
# When each ends: summary.json and a results_b.tsv row, as for the queue's other runs.
# The sealed partition is not read.
#
#     setsid nohup bash resume_paused.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="$REPO/experiments/queue12_2026-09-30"
LOG="$DIR/driver_b.log"
RES="$DIR/resources_b.tsv"
SUMMARIZE="$REPO/experiments/foundational_2026-09-29/summarize_run.py"
PYH="$HOME/tfenv/bin/python"
IMG=qdec:tf215
log () { echo "[queue12 $(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG"; }

declare -A PIDS=() OUTS=()
prepare_and_launch () {   # <paused run name>
  local name="$1" SRC SD N TAG S OUT ENVX
  SRC=$(ls -d "$DIR/${name}"_2* | grep -v resume | head -1); SD="$SRC/seed0"
  N=$(grep -c 'ms/step' "$SD/run.log")
  TAG=$(basename "$(ls "$SD"/ckpt/*.best.json)" .best.json)
  S=$(date -u +%Y%m%dT%H%M%SZ); OUT="$DIR/${name}_resume_$S"
  mkdir -p "$OUT/seed0/ckpt" "$OUT/resume_from"
  cp -p "$SD/ckpt/$TAG".best.{keras,weights.h5,json} "$OUT/seed0/ckpt/"
  cp -p "$SD/ckpt/$TAG.lastepoch.weights.h5" "$OUT/resume_from/"
  "$PYH" - "$SD/run.log" "$N" "$OUT/resume_from/$TAG.history.json" <<'PY'
import json, re, sys
log, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
lines = [l for l in open(log) if 'ms/step' in l]
assert len(lines) == n, (len(lines), n)
h = {}
for l in lines:
    for k, v in re.findall(r' - ([a-z_]+): ([-0-9.e+]+)', l):
        h.setdefault(k, []).append(float(v))
assert all(len(v) == n for v in h.values()), {k: len(v) for k, v in h.items()}
json.dump(h, open(out, 'w'))
print(f'history rebuilt: {n} epochs, keys {sorted(h)}')
PY
  { echo "at_start repo_head $(git -C "$REPO" rev-parse HEAD)"
    echo "resumed_from $SRC (after epoch $N; history epochs 1-$N rebuilt from run.log)"
    (cd "$OUT" && sha256sum resume_from/* seed0/ckpt/*)
  } > "$OUT/code_commit.txt"
  # the original launcher environment, with OUT/STAMP replaced and the resume settings added
  ENVX=()
  while read -r kv; do
    case "$kv" in OUT=*|STAMP=*) ;; *) ENVX+=("$kv") ;; esac
  done < <(python3 - "$SRC/command.txt" <<'PY'
import re, shlex, sys
cmd = open(sys.argv[1]).read()
env = cmd.split(' env ', 1)[1].rsplit(' bash gh200_run_student_scaling.sh', 1)[0]
# EXTRA_POOL and EXTRA_N hold space-separated lists; keep each KEY=VALUE whole
for m in re.finditer(r'([A-Z_]+)=(.*?)(?= [A-Z_]+=|$)', env):
    print(f'{m.group(1)}={m.group(2)}')
PY
)
  ENVX+=(OUT="$OUT" STAMP="$S" MALLOC_ARENA_MAX=2 INITIAL_EPOCH="$N"
         RESUME_WEIGHTS="$OUT/resume_from/$TAG.lastepoch.weights.h5"
         RESUME_BEST_JSON="$OUT/seed0/ckpt/$TAG.best.json"
         RESUME_HISTORY="$OUT/resume_from/$TAG.history.json")
  printf '%s\n' "${ENVX[@]}" > "$OUT/env.txt"
  echo "podman run --rm --name q12_${name}_resume ... env $(printf '%q ' "${ENVX[@]}") bash gh200_run_student_scaling.sh" > "$OUT/command.txt"
  log "START ${name}_resume from epoch $N -> $OUT (host available $(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo) GB)"
  podman run --rm --name "q12_${name}_resume" --security-opt=label=disable --device nvidia.com/gpu=all \
    -v "$HOME:$HOME" -e HOME="$HOME" -w "$REPO/src" "$IMG" \
    env "${ENVX[@]}" bash gh200_run_student_scaling.sh > "$OUT/podman.log" 2>&1 &
  PIDS[$name]=$!; OUTS[$name]=$OUT
  local i
  for i in $(seq 1 240); do
    grep -qE 'ms/step|Error|STOP' "$OUT/seed0/run.log" 2>/dev/null && break
    kill -0 "${PIDS[$name]}" 2>/dev/null || break
    sleep 15
  done
  grep -E 'resumed from|shuffle order advanced|Error|STOP' "$OUT/seed0/run.log" 2>/dev/null | tail -3 | sed "s/^/[${name}] /" >> "$LOG"
}

for r in R12_d7_u280_100M_p005 R08_d5_u280_100M R04_d9_u140_100M; do prepare_and_launch "$r"; done

for r in "${!PIDS[@]}"; do
  wait "${PIDS[$r]}"; OUT="${OUTS[$r]}"; echo $? > "$OUT/exit_code"
  if [ -f "$OUT/seed0/COMPLETE" ] && "$PYH" "$SUMMARIZE" "$OUT" "$RES" "q12_${r}_resume" > "$OUT/seed0/summary.json" 2>> "$DIR/summarize.log"; then
    log "DONE  ${r}_resume: $("$PYH" -c "import json;d=json.load(open('$OUT/seed0/summary.json'));print(' '.join(f'{k}={d.get(k)}' for k in ('params','p_L','mwpm_p_L','ratio','best_epoch','best_val_loss','best_val_ler','final_train_loss','final_val_loss','runtime_h')))")"
    "$PYH" "$DIR/results_row.py" "$OUT" >> "$DIR/results_b.tsv" 2>> "$DIR/summarize.log"
  else
    log "FAIL  ${r}_resume ended without COMPLETE; see $OUT/seed0/run.log"
  fi
done
log "END   resume of paused runs"
