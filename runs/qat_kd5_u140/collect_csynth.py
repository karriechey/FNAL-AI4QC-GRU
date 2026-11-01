# Created: 2026-10-28 | Last updated: 2026-10-28
# csynth.xml of an hls4ml project -> one CSV row (python3 standard library only)
#   python3 collect_csynth.py <project dir> <out.csv>
import csv
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET

prj, out = sys.argv[1], sys.argv[2]
files = glob.glob(os.path.join(prj, '*_prj', 'solution*', 'syn', 'report', 'myproject_csynth.xml'))
if not files:
    sys.exit(f'no myproject_csynth.xml under {prj}')
root = ET.parse(files[0]).getroot()
perf, area = root.find('./PerformanceEstimates'), root.find('./AreaEstimates')
row = {'config': os.path.basename(os.path.dirname(os.path.abspath(prj))),
       'part': root.findtext('./UserAssignments/Part'),
       'target_clock_ns': root.findtext('./UserAssignments/TargetClockPeriod'),
       'est_clock_ns': perf.findtext('./SummaryOfTimingAnalysis/EstimatedClockPeriod'),
       'latency_best_cycles': perf.findtext('./SummaryOfOverallLatency/Best-caseLatency'),
       'latency_worst_cycles': perf.findtext('./SummaryOfOverallLatency/Worst-caseLatency'),
       'ii_min': perf.findtext('./SummaryOfOverallLatency/Interval-min'),
       'ii_max': perf.findtext('./SummaryOfOverallLatency/Interval-max')}
for child in area.find('./Resources'):
    row[child.tag.replace('DSP48E', 'DSP')] = child.text
for child in area.find('./AvailableResources'):
    row['available_' + child.tag.replace('DSP48E', 'DSP')] = child.text
for k in ('versions.json', 'precisions.json'):
    p = os.path.join(os.path.dirname(os.path.abspath(prj)), k)
    if os.path.exists(p):
        row[k.split('.')[0]] = json.dumps(json.load(open(p)))
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(row))
    w.writeheader()
    w.writerow(row)
