#!/usr/bin/env python3
# Created: 2026-11-03 | Last updated: 2026-11-03
# One run folder -> one row in RESULTS_HOME/rows, all rows -> large_parts_results.csv; also tars the job's reports.
#   python3 collect.py        rebuilds everything from runs/
import csv
import glob
import json
import os
import re
import tarfile
import time
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_HOME = os.path.expanduser('~/d7_4x70_large_parts_results')
CLOCK_NS = 5.0
LAYERS = {'2': 'gru', '3': 'gru_1', '4': 'gru_2', '5': 'gru_3', '6': 'logit'}
# hls4ml C-sim, eval block [15.2M, 17M); FP32 pL 0.003636, uncorrelated MWPM 0.004291
PL = {'P8_t4096': (0.184179, 50.65, 42.92), 'P16_tdefault': (0.003663, 1.007, 0.854)}
# used only when the report has no available count
AVAILABLE = {'vu13p': {'LUT': 1728000, 'FF': 3456000, 'DSP': 12288, 'BRAM_18K': 5376, 'URAM': 1280},
             'vu19p': {'LUT': 4086720, 'DSP': 3840},
             'vp1902': {'LUT': 8460288, 'DSP': 6864}}
FIELDS = ['job', 'stage', 'status', 'config', 'weight_bits', 'other_bits', 'tables', 'rf', 'rf_dense', 'n_unit_par', 'n_cycles',
          'backend', 'tool', 'tool_version', 'hls4ml_commit', 'part', 'target_clock_ns', 'est_clock_ns', 'timing_met',
          'latency_min_cycles', 'latency_max_cycles', 'latency_source', 'latency_us_at_5ns', 'ii_min_cycles', 'ii_max_cycles',
          'LUT', 'LUT_available', 'LUT_pct', 'FF', 'FF_available', 'FF_pct', 'DSP', 'DSP_available', 'DSP_pct',
          'BRAM_18K', 'BRAM_18K_available', 'BRAM_18K_pct', 'URAM', 'URAM_available', 'URAM_pct', 'fits_part', 'over_limit',
          'lat_gru', 'lat_gru_1', 'lat_gru_2', 'lat_gru_3', 'lat_logit', 'LUT_gru', 'LUT_gru_1', 'LUT_gru_2', 'LUT_gru_3', 'LUT_logit',
          'csim_100_ok', 'csim_100_max_abs_diff', 'csynth_wall_h', 'csynth_peak_gb', 'host', 'cpu', 'n_cpus', 'start_utc', 'end_utc',
          'pL', 'pL_over_fp32', 'pL_over_mwpm', 'pL_note']


def number(text):
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def module_reports(report_dir, row):
    total = 0
    known = True
    for name, layer in LAYERS.items():
        kind = 'dense' if layer == 'logit' else 'gru_stack'
        files = glob.glob(os.path.join(report_dir, f'{kind}*config{name}_*csynth.xml'))
        if not files:
            known = False
            continue
        root = ET.parse(sorted(files)[0]).getroot()
        latency = number(root.findtext('./PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency'))
        row[f'lat_{layer}'] = latency if latency is not None else 'unknown'
        row[f'LUT_{layer}'] = root.findtext('./AreaEstimates/Resources/LUT')
        if latency is None:
            known = False
        else:
            total += latency
    return total if known else None


def parse_report(work, row, part_key):
    report_dir = os.path.join(work, 'project', 'myproject_prj', 'solution1', 'syn', 'report')
    path = os.path.join(report_dir, 'myproject_csynth.xml')
    if not os.path.exists(path):
        return
    root = ET.parse(path).getroot()
    perf, area = root.find('./PerformanceEstimates'), root.find('./AreaEstimates')
    latency = perf.find('./SummaryOfOverallLatency')
    row.update(target_clock_ns=root.findtext('./UserAssignments/TargetClockPeriod'),
               est_clock_ns=perf.findtext('./SummaryOfTimingAnalysis/EstimatedClockPeriod'),
               latency_min_cycles=latency.findtext('Best-caseLatency'), latency_max_cycles=latency.findtext('Worst-caseLatency'),
               ii_min_cycles=latency.findtext('Interval-min'), ii_max_cycles=latency.findtext('Interval-max'))
    summed = module_reports(report_dir, row)
    cycles = number(row['latency_max_cycles'])
    row['latency_source'] = 'report'
    if cycles is None and summed is not None:
        cycles, row['latency_source'] = summed, 'sum of layers'
    elif cycles is None:
        row['latency_source'] = 'unknown'
    if cycles is not None:
        row['latency_max_cycles'], row['latency_us_at_5ns'] = cycles, round(cycles * CLOCK_NS / 1000, 3)
    try:
        row['timing_met'] = float(row['est_clock_ns']) <= float(row['target_clock_ns'])
    except (TypeError, ValueError):
        row['timing_met'] = ''
    available = {c.tag.replace('DSP48E', 'DSP'): number(c.text) for c in area.find('./AvailableResources')}
    over = []
    for c in area.find('./Resources'):
        key = c.tag.replace('DSP48E', 'DSP')
        limit = available.get(key) or AVAILABLE.get(part_key, {}).get(key)
        row[key], row[f'{key}_available'] = c.text, limit
        if limit and number(c.text) is not None:
            row[f'{key}_pct'] = round(100 * int(c.text) / limit, 2)
            if int(c.text) > limit:
                over.append(key)
    row['fits_part'], row['over_limit'] = not over, ' '.join(over)


def job_row(work):
    meta = json.load(open(os.path.join(work, 'run.json')))
    row = {k: meta.get(k) for k in ('job', 'stage', 'status', 'config', 'rf', 'backend', 'tool', 'tool_version', 'part', 'host', 'cpu',
                                    'n_cpus', 'start_utc', 'end_utc')}
    row['weight_bits'], row['other_bits'] = (8, 16) if meta['config'].startswith('P8') else (16, 16)
    row['tables'] = '4096 x ap_fixed<16,1>' if meta['config'].endswith('t4096') else '1024 x ap_fixed<18,8>'
    row['hls4ml_commit'] = '1259e2b25 with the gru_stack output loop patch'
    row['pL'], row['pL_over_fp32'], row['pL_over_mwpm'] = PL[meta['config']]
    row['pL_note'] = 'full-block C-sim at RF 70 without the patch; RF, backend and patch give identical logits on the first 100,000 shots'
    params = os.path.join(work, 'project', 'firmware', 'parameters.h')
    if os.path.exists(params):
        text = open(params).read()
        cell = text[text.index('struct config2 :'):]
        row['n_unit_par'] = re.search(r'n_unit_par = (\d+)', cell).group(1)
        row['n_cycles'] = re.search(r'n_cycles = (\d+)', cell).group(1)
        row['rf_dense'] = max(int(meta['rf']), 35)
    compare = meta.get('csim_compare', {})
    row['csim_100_ok'], row['csim_100_max_abs_diff'] = compare.get('ok'), compare.get('max_abs_diff')
    synth = meta.get('csynth')
    if synth:
        row['csynth_wall_h'], row['csynth_peak_gb'] = round(synth['wall_s'] / 3600, 3), round(synth['peak_rss_mb'] / 1024, 2)
    parse_report(work, row, meta['part_key'])
    return row


def write_csv():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_HOME, 'rows', '*.json'))):
        rows.append(json.load(open(path)))
    out = os.path.join(RESULTS_HOME, 'large_parts_results.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    return out


def collect_job(work):
    row = job_row(work)
    rows_dir = os.path.join(RESULTS_HOME, 'rows')
    os.makedirs(rows_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    json.dump(row, open(os.path.join(rows_dir, f"{row['job']}__{row['host']}.json"), 'w'), indent=1, default=str)
    prj = os.path.join(work, 'project')
    with tarfile.open(os.path.join(RESULTS_HOME, f"reports_{row['job']}_{row['host']}_{stamp}.tar.gz"), 'w:gz') as tar:
        tar.add(os.path.join(work, 'run.json'), arcname=f"{row['job']}/run.json")
        tar.add(os.path.join(work, 'logs'), arcname=f"{row['job']}/logs")
        for name in glob.glob(os.path.join(prj, 'myproject_prj', 'solution1', 'syn', 'report', '*.rpt')) + \
                glob.glob(os.path.join(prj, 'myproject_prj', 'solution1', 'syn', 'report', '*.xml')):
            tar.add(name, arcname=f"{row['job']}/report/{os.path.basename(name)}")
        for name in ('firmware/parameters.h', 'firmware/defines.h', 'hls4ml_config_used.json', 'project.tcl', 'tb_data/csim_results.log'):
            if os.path.exists(os.path.join(prj, name)):
                tar.add(os.path.join(prj, name), arcname=f"{row['job']}/{name}")
    out = write_csv()
    keys = ('job', 'status', 'part', 'rf', 'latency_max_cycles', 'latency_us_at_5ns', 'ii_max_cycles', 'est_clock_ns', 'LUT', 'LUT_pct',
            'DSP', 'DSP_pct', 'BRAM_18K', 'BRAM_18K_pct', 'over_limit', 'csynth_wall_h')
    return f"{ {k: row.get(k) for k in keys} } -> {out}"


if __name__ == '__main__':
    for run in sorted(glob.glob(os.path.join(HERE, 'runs', '*', 'run.json'))):
        print(collect_job(os.path.dirname(run)))
