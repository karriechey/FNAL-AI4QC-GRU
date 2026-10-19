#!/usr/bin/env bash
# Created: 2026-10-18
# Last modified: 2026-10-18
#   bash run_teacher_check.sh 5|7|9
# TF32 rule: train with TF32 on only if its eval-block error count is within 5 of --no-tf32.
set -uo pipefail
D=$1
source "$(dirname "$0")/teachers.sh"
SRC="$(cd "$PILOT/../../src" && pwd)"
mkdir -p "$G0" "$PILOT/teachers"
[ -f "$T_COPY" ] || cp -p "$T_SRC" "$T_COPY"
[ "$(sha256sum < "$T_SRC")" = "$(sha256sum < "$T_COPY")" ] || { echo "FAIL d$D: teacher copy differs from $T_SRC"; exit 1; }
read -r -a TU <<< "$T_UNITS"
HOSTPY=${HOSTPY:-$HOME/tfenv/bin/python}

run () {   # container name, command...
  local name=$1; shift
  podman run --rm --name "$name" --security-opt=label=disable --device nvidia.com/gpu=all \
    -v "$HOME:$HOME" -e HOME="$HOME" -e TF_DETERMINISTIC_OPS=1 -e TF_CUDNN_DETERMINISTIC=1 \
    -e MALLOC_ARENA_MAX=2 -w "$SRC" qdec:tf215 "$@"
}

DUMP=$G0/teacher_eval_per_shot.npz
if [ ! -f "$DUMP" ]; then
  run "tcheck_eval_d$D" python3 eval_student_on_tail.py --student gru --inputs evts --hidden \
    --units "${TU[@]}" --d "$D" --p 0.004 --rounds "$D" --pool "$POOL" --weights "$T_COPY" \
    --eval-start 15200000 --n-test 1800000 --batch-size 10000 --no-tf32 --gpu-mem-mib 8000 \
    --dump-per-shot "$DUMP" --out-csv "$G0/teacher_eval.csv" > "$G0/teacher_eval.log" 2>&1 \
    || { echo "FAIL d$D: eval_student_on_tail.py, see $G0/teacher_eval.log"; exit 1; }
fi
for mode in off on; do
  [ "$mode" = off ] && X=(--no-tf32 --n 100000) || X=(--n 1800000)
  run "tcheck_${mode}_d$D" python3 "$PILOT/teacher_check.py" --d "$D" --units "${TU[@]}" \
    --weights "$T_COPY" --params "$T_PARAMS" --epoch "$T_EPOCH" --pool "$POOL" --dump "$DUMP" \
    "${X[@]}" --out-json "$G0/check_tf32_$mode.json" > "$G0/check_tf32_$mode.log" 2>&1 \
    || { echo "FAIL d$D: teacher_check.py tf32 $mode, see $G0/check_tf32_$mode.log"; exit 1; }
done

"$HOSTPY" - "$G0" "$T_EVAL_ERR" "$T_PARAMS" "$T_EPOCH" <<'PY'
import json, os, sys
import numpy as np
g0, logged, params, epoch = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
dump = np.load(os.path.join(g0, 'teacher_eval_per_shot.npz'))
eval_err = int((~dump['student_correct'].astype(bool)).sum())
off = json.load(open(os.path.join(g0, 'check_tf32_off.json')))
on = json.load(open(os.path.join(g0, 'check_tf32_on.json')))
checks = {
    f'eval block errors {eval_err} == logged {logged}': eval_err == logged,
    f'parameters {off["params"]} == {params}': off['params'] == params,
    f'epoch {off["epoch"]} == {epoch}': off['epoch'] == epoch,
    'training layout features == eval features': off['features_equal'],
    'labels equal to the dump': off['labels_equal'] and on['labels_equal'],
    f'100k training path: errors {off["errors_training_path"]} == dump {off["errors_eval_dump"]}, '
    f'{off["decisions_differ"]} decisions differ, max |logit diff| {off["max_abs_logit_diff"]:.2e}':
        off['errors_training_path'] == off['errors_eval_dump'] and off['decisions_differ'] == 0,
}
gap = on['errors_training_path'] - on['errors_eval_dump']
decision = 'on' if abs(gap) <= 5 else 'off'
open(os.path.join(g0, 'tf32_decision.txt'), 'w').write(decision + '\n')
ok = all(checks.values())
summary = dict(passed=ok, checks=checks, eval_errors=eval_err, tf32_errors=on['errors_training_path'],
               tf32_gap=gap, tf32_decisions_differ=on['decisions_differ'],
               tf32_max_abs_logit_diff=on['max_abs_logit_diff'], train_tf32=decision, sha256=off['sha256'])
json.dump(summary, open(os.path.join(g0, 'summary.json'), 'w'), indent=1)
for k, v in checks.items():
    print(('PASS ' if v else 'FAIL ') + k)
print(f'TF32 on, whole eval block: {on["errors_training_path"]} errors vs {eval_err} (gap {gap:+d}, '
      f'{on["decisions_differ"]} decisions differ) -> train with TF32 {decision}')
print('PASS' if ok else 'FAIL', f'd{sys.argv[1][-1]}')
sys.exit(0 if ok else 1)
PY
