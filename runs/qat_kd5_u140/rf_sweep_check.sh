#!/usr/bin/env bash
# Created: 2026-10-29 | Last updated: 2026-10-29
# Candidate w12_s6 at RF 10, 35, 140: Vivado projects, C-sim vs the verified RF 70 logits (100k shots), exports
cd "$(dirname "$0")"
for rf in 10 35 140; do
  bash hpy.sh p3_vivado.py --n 100000 --rf $rf --workers 48 qat:w12_s6_seed0 > logs/p3_vivado_rf$rf.log 2>&1
  echo "$(date -u +%FT%TZ) rf $rf exit $?" >> logs/rf_sweep_check.log
done
