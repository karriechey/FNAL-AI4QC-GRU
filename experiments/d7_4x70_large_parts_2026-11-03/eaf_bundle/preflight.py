#!/usr/bin/env python3
# Created: 2026-11-03 | Last updated: 2026-11-04
# Reports HLS tools and versions, and whether each target part is present and licensed, by running a tiny
# C-sim + C synthesis per tool and part. Takes minutes. Writes preflight/ and copies the report to RESULTS_HOME.
#   python3 preflight.py
import csv
import glob
import os
import platform
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'preflight')
RESULTS_HOME = os.path.expanduser('~/d7_4x70_large_parts_results')
ROOTS = ['/opt/xilinx', '/opt/Xilinx', '/tools/Xilinx', '/tools/xilinx', '/usr/local/xilinx', '/data/xilinx']
ROOTS += [r for r in os.environ.get('XILINX_ROOTS', '').split(':') if r]
HLS_TOOLS = ['vivado_hls', 'vitis_hls', 'vitis-run']
PARTS = {'vu13p': ('xcvu13p', ['xcvu13p-flga2577-2-e']),
         'vu19p': ('xcvu19p', ['xcvu19p-fsvb3824-2-e', 'xcvu19p-fsva3824-2-e']),
         'vp1902': ('xcvp1902', ['xcvp1902-vsva6865-2MP-e-S'])}
LICENSE = re.compile(r'valid license was not found|\[Common 17-345\]|failed to get a license|license checkout|'
                     r'no such feature|license (file|server).*(not|fail)', re.I)
PART = re.compile(r'(ERROR|CRITICAL).*\b(part|device)\b|HLS 200-1023|is not installed', re.I)

PROBE_CPP = '''#include "ap_fixed.h"
void probe(ap_fixed<16, 6> a[4], ap_fixed<8, 3> b[4], ap_fixed<24, 10> &out) {
    #pragma HLS PIPELINE II=1
    #pragma HLS ARRAY_PARTITION variable=a complete
    #pragma HLS ARRAY_PARTITION variable=b complete
    ap_fixed<24, 10> sum = 0;
    for (int i = 0; i < 4; i++) sum += a[i] * b[i];
    out = sum;
}
'''
PROBE_TB = '''#include "ap_fixed.h"
void probe(ap_fixed<16, 6> a[4], ap_fixed<8, 3> b[4], ap_fixed<24, 10> &out);
int main() {
    ap_fixed<16, 6> a[4] = {1, 2, 3, 4};
    ap_fixed<8, 3> b[4] = {1, 1, 1, 1};
    ap_fixed<24, 10> out;
    probe(a, b, out);
    return out == 10 ? 0 : 1;
}
'''
PROBE_V = 'module probe_v(input clk, input [7:0] a, output reg [7:0] b);\nalways @(posedge clk) b <= a + 1;\nendmodule\n'


def shell(settings, command, timeout=1200, cwd=None):
    prefix = f'source "{settings}" > /dev/null 2>&1; ' if settings else ''
    try:
        p = subprocess.run(['bash', '-c', prefix + command], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=timeout)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired as e:
        return 124, str(e.stdout or '') + '\nTIMEOUT'


def find_toolchains():
    settings = ['']
    for root in ROOTS:
        settings += glob.glob(f'{root}/*/*/settings64.sh') + glob.glob(f'{root}/*/settings64.sh')
    if os.environ.get('VIVADO_SETTINGS'):
        settings.append(os.environ['VIVADO_SETTINGS'])
    chains, seen = [], set()
    for s in sorted(set(settings)):
        rc, out = shell(s, 'for t in vivado_hls vitis_hls vitis-run vivado; do command -v $t; done', 60)
        for path in out.split():
            tool = os.path.basename(path)
            if path.startswith('/') and path not in seen:
                seen.add(path)
                rc, version = shell(s, f'{tool} {"--version" if tool == "vitis-run" else "-version"}', 300)
                m = re.search(r'20\d\d\.\d', version)
                chains.append({'tool': tool, 'version': m.group(0) if m else 'unknown', 'settings': s, 'path': path})
    return chains


def hls_command(tool, tcl):
    return f'vitis-run --mode hls --tcl {tcl}' if tool == 'vitis-run' else f'{tool} -f {tcl}'


def classify(log, rc, part_seen):
    if rc == 124:
        return 'OTHER', 'probe timed out'
    m = LICENSE.search(log)
    if m:
        return 'LICENSE', next(l for l in log.splitlines() if m.group(0) in l).strip()[:300]
    if part_seen:
        return 'PASS', ''
    for line in log.splitlines():
        if PART.search(line):
            return 'PART', line.strip()[:300]
    errors = [l.strip() for l in log.splitlines() if 'ERROR' in l]
    return 'OTHER', (errors[0] if errors else f'rc {rc}, no csynth report')[:300]


def probe(chain, part):
    work = os.path.join(OUT, 'probes', f"{chain['tool']}_{chain['version']}", part)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    open(os.path.join(work, 'probe.cpp'), 'w').write(PROBE_CPP)
    open(os.path.join(work, 'tb.cpp'), 'w').write(PROBE_TB)
    open(os.path.join(work, 'p.tcl'), 'w').write(
        'open_project -reset prj\nset_top probe\nadd_files probe.cpp\nadd_files -tb tb.cpp\n'
        f'open_solution -reset s\nset_part {{{part}}}\ncreate_clock -period 5\ncsim_design\ncsynth_design\nexit\n')
    t0 = time.time()
    rc, log = shell(chain['settings'], hls_command(chain['tool'], 'p.tcl'), cwd=work)
    open(os.path.join(work, 'probe.log'), 'w').write(log)
    xml = glob.glob(os.path.join(work, 'prj', 's', 'syn', 'report', 'probe_csynth.xml'))
    row = dict(tool=chain['tool'], tool_version=chain['version'], settings=chain['settings'], part=part, wall_s=int(time.time() - t0),
               log=os.path.relpath(os.path.join(work, 'probe.log'), HERE))
    part_seen = False
    if xml:
        root = ET.parse(xml[0]).getroot()
        part_seen = part.lower() in (root.findtext('./UserAssignments/Part') or '').lower()
        res = root.find('./AreaEstimates/Resources')
        row.update(probe_lut=res.findtext('LUT'), probe_ff=res.findtext('FF'),
                   probe_dsp=res.findtext('DSP') or res.findtext('DSP48E'))
    row['status'], row['reason'] = classify(log, rc, part_seen)
    return row


def list_parts(vivado_chains, prefix):
    found = []
    for chain in vivado_chains:
        work = os.path.join(OUT, 'probes', f"vivado_{chain['version']}")
        os.makedirs(work, exist_ok=True)
        tcl = os.path.join(work, f'parts_{prefix}.tcl')
        open(tcl, 'w').write(f'foreach p [get_parts -quiet {prefix}*] {{puts "PARTNAME $p"}}\nexit\n')
        _, out = shell(chain['settings'], f'vivado -mode batch -nojournal -nolog -source {tcl}', cwd=work)
        found += re.findall(r'^PARTNAME (\S+)', out, re.M)
    return sorted(set(found))


def implementation_license(chain, part):
    # C synthesis does not need this license; place and route later does
    work = os.path.join(OUT, 'probes', f"vivado_{chain['version']}", part)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    open(os.path.join(work, 'probe.v'), 'w').write(PROBE_V)
    open(os.path.join(work, 's.tcl'), 'w').write(f'read_verilog probe.v\nsynth_design -top probe_v -part {part}\nexit\n')
    rc, log = shell(chain['settings'], 'vivado -mode batch -nojournal -nolog -source s.tcl', cwd=work)
    open(os.path.join(work, 'synth.log'), 'w').write(log)
    if LICENSE.search(log):
        return 'missing'
    if PART.search(log):
        return 'part_not_in_vivado'
    return 'ok' if rc == 0 else 'unknown'


def version_key(row):
    return row['tool_version']


def select(rows):
    chosen = {}
    for key in PARTS:
        passed = [r for r in rows if r['part_key'] == key and r['status'] == 'PASS']
        vivado = [r for r in passed if r['tool'] == 'vivado_hls']
        if key.startswith('vu') and vivado:
            chosen[key] = vivado[0]
        elif passed:
            chosen[key] = max(passed, key=version_key)
    return chosen


def failure_message(key, rows, chains):
    target = PARTS[key][0]
    hls = [c for c in chains if c['tool'] in HLS_TOOLS]
    if not hls:
        return f'TOOL: no HLS tool found (searched PATH and {", ".join(ROOTS)}).'
    mine = [r for r in rows if r['part_key'] == key]
    licensed = [r for r in mine if r['status'] == 'LICENSE']
    if licensed:
        r = licensed[0]
        return f"LICENSE: {r['tool']} {r['tool_version']} has {r['part']} but no license for it. Log line: {r['reason']}"
    if key == 'vp1902' and not any(c['tool'] != 'vivado_hls' for c in hls):
        versions = ', '.join(f"{c['tool']} {c['version']}" for c in hls)
        return f'TOOL: only {versions} here. VP1902 needs Vitis HLS with Versal Premium device files installed.'
    if all(r['status'] == 'PART' for r in mine):
        tried = ', '.join(sorted({f"{r['tool']} {r['tool_version']}" for r in mine}))
        return f'PART: {target} is not installed in {tried}. Tried {", ".join(sorted({r["part"] for r in mine}))}.'
    r = next((r for r in mine if r['status'] == 'OTHER'), None)
    return f"OTHER: {r['tool']} {r['tool_version']} on {r['part']}: {r['reason']}. See {r['log']}." if r else 'OTHER: no probe ran.'


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    chains = find_toolchains()
    hls = [c for c in chains if c['tool'] in HLS_TOOLS]
    # vitis-run and vitis_hls of one install are the same HLS engine
    versions_with_vitis_hls = {c['version'] for c in hls if c['tool'] == 'vitis_hls'}
    hls = [c for c in hls if not (c['tool'] == 'vitis-run' and c['version'] in versions_with_vitis_hls)]
    vivado = [c for c in chains if c['tool'] == 'vivado']
    rows, discovered = [], {}
    for key, (prefix, candidates) in PARTS.items():
        discovered[key] = list_parts(vivado, prefix)
        try_parts = candidates + [p for p in discovered[key] if p not in candidates][:2]
        for chain in hls:
            for part in try_parts:
                row = probe(chain, part)
                row.update(part_key=key, impl_license='')
                rows.append(row)
                print(key, chain['tool'], chain['version'], part, row['status'], row['reason'], flush=True)
                if row['status'] == 'PASS':
                    break
    for row in rows:
        if row['status'] == 'PASS' and vivado:
            same = [c for c in vivado if c['version'] == row['tool_version']] or vivado
            row['impl_license'] = implementation_license(same[0], row['part'])

    fields = ['part_key', 'part', 'tool', 'tool_version', 'status', 'reason', 'impl_license', 'probe_lut', 'probe_ff',
              'probe_dsp', 'wall_s', 'settings', 'log']
    with open(os.path.join(OUT, 'preflight.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    chosen = select(rows)
    with open(os.path.join(OUT, 'selected.tsv'), 'w') as f:
        for key, r in chosen.items():
            f.write('\t'.join((key, r['part'], r['tool'], r['tool_version'], r['settings'])) + '\n')

    lines = [f'host {platform.node()}  cpus {os.cpu_count()}',
             f"license env XILINXD_LICENSE_FILE={os.environ.get('XILINXD_LICENSE_FILE', '')} "
             f"LM_LICENSE_FILE={os.environ.get('LM_LICENSE_FILE', '')}", '', 'Tools found:']
    lines += [f"  {c['tool']} {c['version']}  {c['path']}" for c in chains] or ['  none']
    lines += ['', 'Parts:']
    for key in PARTS:
        lines.append(f"  {key}: Vivado part list: {', '.join(discovered[key]) or 'no Vivado or no match'}")
        if key in chosen:
            r = chosen[key]
            exact = r['part'] in PARTS[key][1]
            lines.append(f"    USABLE with {r['tool']} {r['tool_version']}, part {r['part']}"
                         f"{'' if exact else ' (fallback: speed grade or package differs from the target)'}, "
                         f"implementation license {r['impl_license'] or 'not checked'}")
        else:
            lines.append(f'    NOT USABLE. {failure_message(key, rows, chains)}')
    usable = sorted(chosen)
    jobs = [l.split('\t') for l in open(os.path.join(HERE, 'jobs.tsv')).read().splitlines()[1:]]
    lines += ['', 'Jobs that can run: ' + (' '.join(j[0] for j in jobs if j[2] in usable) or 'none')]
    report = '\n'.join(lines) + '\n'
    open(os.path.join(OUT, 'preflight_report.txt'), 'w').write(report)
    os.makedirs(RESULTS_HOME, exist_ok=True)
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    shutil.copy(os.path.join(OUT, 'preflight_report.txt'), os.path.join(RESULTS_HOME, f'preflight_report_{stamp}.txt'))
    shutil.copy(os.path.join(OUT, 'preflight.csv'), os.path.join(RESULTS_HOME, f'preflight_{stamp}.csv'))
    print('\n' + report)


main()
