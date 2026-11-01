#!/usr/bin/env bash
# Created: 2026-11-01 | Last updated: 2026-11-01
# Unattended chain after the alpha x T sweep: wait for it, rank it, pick the round-2 teacher and (alpha, T)
# (choose.py), check that teacher, train the round-2 students, report. Progress in STATUS.
#   setsid nohup bash after_sweep.sh > logs/after_sweep.out 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SWEEP="$(cd "$HERE/../kd_sweep_d5_u140_2026-10-31" && pwd)"
HOSTPY=${HOSTPY:-$HOME/tfenv/bin/python}
WIDTHS=(70)                    # 20,021 parameters; one bet, see README
mkdir -p "$HERE/logs"
status () { echo "$(date -u +%FT%TZ) $*" | tee -a "$HERE/STATUS"; }

n_new=$(grep -v '^#' "$SWEEP/configs.tsv" | awk -F'\t' '$4 == "new"' | wc -l)
status "waiting for the sweep ($n_new new cells)"
while :; do
  done_n=$(ls -d "$SWEEP"/keep/*/DONE 2>/dev/null | wc -l)
  [ "$done_n" -ge "$n_new" ] && break
  grep -q "all cells done or given up" "$SWEEP/logs/queue.out" 2>/dev/null \
    && ! pgrep -f "kd_sweep_d5_u140_2026-10-31/run_one.sh" > /dev/null \
    && { status "sweep queue finished with $done_n of $n_new cells done; continuing with those"; break; }
  sleep 600
done
corr=$SWEEP/mwpm_correlated_per_shot_d5_n1800000.npz
( cd "$SWEEP" && "$HOSTPY" report.py $([ -f "$corr" ] && echo --correlated "$corr") ) > "$HERE/logs/sweep_report.txt" 2>&1 \
  || { status "FAIL sweep report.py, see logs/sweep_report.txt"; exit 1; }
status "sweep ranked: $(sed -n 3p "$SWEEP/results/sweep_table.md")"
"$HOSTPY" "$HERE/choose.py" "$SWEEP/results/sweep_results.json" > "$HERE/logs/choose.txt" 2>&1 \
  || { status "FAIL choose.py"; exit 1; }
get () { "$HOSTPY" -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$HERE/choice.json" "$1"; }
if [ "$(get round2)" != True ]; then
  status "best $(get best_run) at $(get best_eval_x_mwpm) x MWPM does not beat MWPM: no round 2"
  exit 0
fi
status "best $(get best_run) ($(get best_eval_x_mwpm) x MWPM); round 2 from $(get teacher_run), alpha $(get r2_alpha) T $(get r2_T)"
bash "$HERE/ta_check.sh" > "$HERE/logs/ta_check.txt" 2>&1 || { status "FAIL teacher check, see logs/ta_check.txt"; exit 1; }
status "teacher check passed ($(tail -1 "$HERE/logs/ta_check.txt"))"

A=$(get r2_alpha); T=$(get r2_T)
for arm in ta; do                # control: the pilot's kd5_u70 (f5w_p004 teacher, alpha 0, T 1, same shots)
  for u in "${WIDTHS[@]}"; do
    setsid nohup bash "$HERE/run_r2.sh" "$arm" "$u" > "$HERE/logs/${arm}_u$u.out" 2>&1 < /dev/null &
    status "started $arm u$u"
    sleep 120
  done
done
t0=$(date +%s)
while pgrep -f "kd_round2_d5_2026-11-01/run_r2.sh" > /dev/null; do sleep 600; done
hours=$(( ($(date +%s) - t0) / 3600 ))
"$HOSTPY" "$HERE/report_r2.py" > "$HERE/logs/report_r2.txt" 2>&1 || { status "FAIL report_r2.py"; exit 1; }
status "round 2 u70 done in ${hours} h: results/r2_table.md"

# trade-off batch: u34 / u48 / u96 together, only if u70 beat MWPM and took at most MAX_H hours
beat=$("$HOSTPY" -c "import json,sys; print(any(r['arm'] == 'ta' and r['units'] == 70 and r['beats_mwpm'] for r in json.load(open(sys.argv[1]))['rows']))" "$HERE/results/r2_results.json")
if [ "$beat" = True ] && [ "$hours" -le "${MAX_H:-8}" ]; then
  for u in 34 48 96; do
    setsid nohup bash "$HERE/run_r2.sh" ta "$u" > "$HERE/logs/ta_u$u.out" 2>&1 < /dev/null &
    status "started trade-off ta u$u"
    sleep 120
  done
  while pgrep -f "kd_round2_d5_2026-11-01/run_r2.sh" > /dev/null; do sleep 600; done
  "$HOSTPY" "$HERE/report_r2.py" > "$HERE/logs/report_r2.txt" 2>&1 || { status "FAIL report_r2.py"; exit 1; }
  status "trade-off batch done: results/r2_table.md"
else
  status "no trade-off batch (u70 beats MWPM: $beat, ${hours} h)"
fi
