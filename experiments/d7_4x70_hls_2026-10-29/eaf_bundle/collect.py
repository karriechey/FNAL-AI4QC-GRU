# Created: 2026-10-29 | Last updated: 2026-10-29
# csynth.xml of every project -> results_table.csv (latency, II, LUT, FF, DSP, BRAM_18K, URAM and their %
# of the device from the same report), plus C-sim check and Vivado wall time / sampled peak memory.
import csv
import glob
import json
import os
import xml.etree.ElementTree as ET

rows = []
for prj in sorted(glob.glob('projects/*')):
    name = os.path.basename(prj)
    row = {'project': name}
    for step in ('csim', 'csynth'):
        f = f'logs/{name}_{step}_cost.json'
        if os.path.exists(f):
            c = json.load(open(f))
            row[f'{step}_rc'], row[f'{step}_wall_s'], row[f'{step}_peak_rss_mb'] = c['rc'], c['wall_s'], c['peak_rss_mb_sampled']
    f = f'logs/{name}_csim_compare.json'
    if os.path.exists(f) and os.path.getsize(f):
        c = json.load(open(f))
        row['csim_100_ok'], row['csim_100_max_abs_diff'] = c.get('ok'), c.get('max_abs_diff')
    xmls = glob.glob(f'{prj}/myproject_prj/solution1/syn/report/myproject_csynth.xml')
    if xmls:
        root = ET.parse(xmls[0]).getroot()
        perf, area = root.find('./PerformanceEstimates'), root.find('./AreaEstimates')
        row.update({'part': root.findtext('./UserAssignments/Part'),
                    'target_clock_ns': root.findtext('./UserAssignments/TargetClockPeriod'),
                    'est_clock_ns': perf.findtext('./SummaryOfTimingAnalysis/EstimatedClockPeriod'),
                    'latency_min_cycles': perf.findtext('./SummaryOfOverallLatency/Best-caseLatency'),
                    'latency_max_cycles': perf.findtext('./SummaryOfOverallLatency/Worst-caseLatency'),
                    'ii_min_cycles': perf.findtext('./SummaryOfOverallLatency/Interval-min'),
                    'ii_max_cycles': perf.findtext('./SummaryOfOverallLatency/Interval-max')})
        avail = {c.tag.replace('DSP48E', 'DSP'): c.text for c in area.find('./AvailableResources')}
        for c in area.find('./Resources'):
            k = c.tag.replace('DSP48E', 'DSP')
            row[k] = c.text
            row[f'{k}_available'] = avail.get(k)
            row[f'{k}_pct'] = round(100 * int(c.text) / int(avail[k]), 2) if avail.get(k) and int(avail[k]) else ''
    rows.append(row)
keys = []
for r in rows:
    keys += [k for k in r if k not in keys]
with open('results_table.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(r['project'], {k: r.get(k) for k in ('latency_max_cycles', 'ii_max_cycles', 'LUT', 'LUT_pct', 'FF', 'FF_pct',
                                               'DSP', 'DSP_pct', 'BRAM_18K', 'BRAM_18K_pct', 'est_clock_ns', 'csim_100_ok')})
