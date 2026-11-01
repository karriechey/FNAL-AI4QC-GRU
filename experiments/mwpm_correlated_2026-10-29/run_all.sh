#!/usr/bin/env bash
# Created: 2026-10-29 | Last updated: 2026-10-29
# Full eval block (1.8M shots) at d = 5, 7, 9, one after another, CPU only.
cd "$(dirname "$0")"
TC=$HOME/FNAL-AI4QC-GRU-kd/experiments/kd_pilot_p004_2026-10/teacher_check
for spec in "5 f5w_p004" "7 f7w_p004" "9 f3280e240"; do
  set -- $spec
  /usr/bin/time -v ~/tfenv/bin/python compare_mwpm.py --d $1 --dump $TC/d$1/teacher_eval_per_shot.npz --gru $2 \
    > logs/full_d$1.log 2> logs/full_d$1.time
  echo "$(date -u +%FT%TZ) d=$1 exit $?" >> logs/run_all.log
done
