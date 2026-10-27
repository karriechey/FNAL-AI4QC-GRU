#!/usr/bin/env python3
# Created: 2026-10-23
# Last modified: 2026-10-27 (several --log parts, for a run resumed more than once)
# Rebuild a run's history.json from its run.log epoch lines, for --resume-history after the
# 2026-10-22 reboot killed six runs before train_student.py wrote one.
#   history_from_log.py --log run_part1.log --epochs 43 --out resume_history.json
# A run resumed more than once has one part log per stretch; pass them in epoch order and the
# rows are concatenated.
#   history_from_log.py --log run_part1.log --log run_part2.log --epochs 59 --out ...
import argparse
import json
import re

EPOCH_LINE = re.compile(r'^\d+/\d+ - \d+s - (.*?) - \d+s/epoch')
METRIC = re.compile(r'([A-Za-z_]+): ([0-9.eE+-]+)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True, action='append')
    ap.add_argument('--epochs', type=int, required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rows = []
    for log in args.log:
        rows += [dict((k, float(v)) for k, v in METRIC.findall(m.group(1)))
                 for m in map(EPOCH_LINE.match, open(log)) if m]
    if len(rows) != args.epochs:
        raise SystemExit(f'[history] {", ".join(args.log)} has {len(rows)} epoch lines, '
                         f'--epochs is {args.epochs}. STOP.')
    keys = set(rows[0])
    if any(set(r) != keys for r in rows):
        raise SystemExit('[history] epoch lines carry different metrics. STOP.')
    json.dump({k: [r[k] for r in rows] for k in sorted(keys)}, open(args.out, 'w'))
    print(f'[history] {args.epochs} epochs, {len(keys)} metrics -> {args.out}')


if __name__ == '__main__':
    main()
