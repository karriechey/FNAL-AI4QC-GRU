#!/usr/bin/env bash
# Created: 2026-10-28 | Last updated: 2026-10-28
# waits for the Phase 3 run to end, then rebuilds results/all_points.csv, qat_seeds.csv and the figures
cd "$(dirname "$0")"
while pgrep -f "bin/python p3_hls[.]py" > /dev/null; do sleep 120; done
bash hpy.sh summarize.py > logs/summarize_after_p3.log 2>&1
date -u +%FT%TZ > results/AFTER_P3_DONE
