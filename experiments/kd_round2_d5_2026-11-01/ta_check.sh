#!/usr/bin/env bash
# Created: 2026-11-01 | Last updated: 2026-11-01
# Teacher check for the round-2 teacher (the pilot's run_teacher_check.sh rules): copy and sha256, eval-block
# errors of its sweep dump equal to the sweep table, training-step path equal to that dump on 100k shots,
# TF32 decision from the whole eval block.
#   bash ta_check.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PILOT="$(cd "$HERE/../kd_pilot_p004_2026-10" && pwd)"
SRC="$(cd "$HERE/../../src" && pwd)"
HOSTPY=${HOSTPY:-$HOME/tfenv/bin/python}
D=5
source "$PILOT/teachers.sh"
source "$HERE/ta_teacher.sh"
mkdir -p "$G0" "$HERE/teachers"
[ -f "$T_COPY" ] || cp -p "$T_SRC" "$T_COPY"
[ "$(sha256sum < "$T_SRC")" = "$(sha256sum < "$T_COPY")" ] || { echo "FAIL: teacher copy differs from $T_SRC"; exit 1; }
cp -p "$T_DUMP" "$G0/teacher_eval_per_shot.npz"

for mode in off on; do
  [ "$mode" = off ] && X=(--no-tf32 --n 100000) || X=(--n 1800000)
  podman run --rm --name "r2check_$mode" --security-opt=label=disable --device nvidia.com/gpu=all \
    -v "$HOME:$HOME" -v /tmp/kchey_kdsweep:/tmp/kchey_kdsweep -e HOME="$HOME" -e TF_DETERMINISTIC_OPS=1 \
    -e TF_CUDNN_DETERMINISTIC=1 -e MALLOC_ARENA_MAX=2 -w "$SRC" qdec:tf215 \
    python3 "$PILOT/teacher_check.py" --d 5 --units $T_UNITS --weights "$T_COPY" --params "$T_PARAMS" \
    --epoch "$T_EPOCH" --pool "$POOL" --dump "$G0/teacher_eval_per_shot.npz" "${X[@]}" \
    --out-json "$G0/check_tf32_$mode.json" > "$G0/check_tf32_$mode.log" 2>&1 \
    || { echo "FAIL: teacher_check.py tf32 $mode, see $G0/check_tf32_$mode.log"; exit 1; }
done
# same summary and TF32 rule as the pilot (the heredoc of run_teacher_check.sh)
sed -n '/^"\$HOSTPY" - /,/^PY$/p' "$PILOT/run_teacher_check.sh" | sed '1d;$d' > "$G0/summarize.py"
"$HOSTPY" "$G0/summarize.py" "$G0" "$T_EVAL_ERR" "$T_PARAMS" "$T_EPOCH"
