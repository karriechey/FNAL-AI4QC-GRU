#!/usr/bin/env python3
# Created: 2026-11-03 | Last updated: 2026-11-03
# Per job: copy the project, set the part chosen by preflight.py, 100-shot C-sim against the grace1 hls4ml C-sim,
# then C synthesis. Results go into the one CSV in RESULTS_HOME, which every pod shares.
#   python3 run_synth.py vu13p_P8_t4096_rf70 [more jobs]    or    python3 run_synth.py --stage A
import argparse
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import time

import collect

HERE = os.path.dirname(os.path.abspath(__file__))
TIMEOUT_H = float(os.environ.get('JOB_TIMEOUT_H', 144))     # scratch deletes files after 7 days
OPT = 'array set opt {{\n reset {reset}\n csim {csim}\n synth {synth}\n cosim 0\n validation 0\n export 0\n vsynth 0\n fifo_opt 0\n pnr 0\n}}\n'


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def say(text):
    print(f'{now()} {text}', flush=True)


def cpu_model():
    try:
        return next(l.split(':')[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name'))
    except (OSError, StopIteration):
        return platform.processor()


def run_tool(sel, prj, step, csim, synth, reset, log_path, timeout_s):
    open(os.path.join(prj, 'build_opt.tcl'), 'w').write(OPT.format(reset=reset, csim=csim, synth=synth))
    cmd = 'vitis-run --mode hls --tcl build_prj.tcl' if sel['tool'] == 'vitis-run' else f"{sel['tool']} -f build_prj.tcl"
    prefix = f"source \"{sel['settings']}\" > /dev/null 2>&1; " if sel['settings'] else ''
    t0, peak = time.time(), 0
    with open(log_path, 'w') as log:
        p = subprocess.Popen(['bash', '-c', prefix + cmd], cwd=prj, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        timed_out = False
        while p.poll() is None:
            out = subprocess.run(['ps', '-o', 'rss=', '-g', str(p.pid)], stdout=subprocess.PIPE, universal_newlines=True).stdout
            peak = max(peak, sum(int(x) for x in out.split()) // 1024)
            if time.time() - t0 > timeout_s:
                timed_out = True
                os.killpg(p.pid, signal.SIGTERM)
                break
            time.sleep(15)
        p.wait()
    return {'step': step, 'rc': 124 if timed_out else p.returncode, 'wall_s': int(time.time() - t0), 'peak_rss_mb': peak}


def run_job(job, sel):
    work = os.path.join(HERE, 'runs', job['job'])
    logs = os.path.join(work, 'logs')
    meta_path = os.path.join(work, 'run.json')
    if os.path.exists(meta_path) and json.load(open(meta_path)).get('status') == 'ok':
        say(f"{job['job']} already done, skipped")
        return
    backend = 'vivado' if sel['tool'] == 'vivado_hls' else 'vitis'
    src = os.path.join(HERE, 'projects', f"{backend}_{job['config']}_rf{job['rf']}")
    meta = dict(job, backend=backend, tool=sel['tool'], tool_version=sel['tool_version'], part=sel['part'], host=platform.node(),
                cpu=cpu_model(), n_cpus=os.cpu_count(), start_utc=now(), status='running')
    if not os.path.isdir(src):
        say(f"{job['job']}: no {backend} project for this job ({os.path.basename(src)}), skipped")
        os.makedirs(logs, exist_ok=True)
        meta['status'] = 'no_project_for_tool'
        json.dump(meta, open(meta_path, 'w'), indent=1)
        return
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(logs)
    prj = os.path.join(work, 'project')
    shutil.copytree(src, prj)
    tcl_path = os.path.join(prj, 'project.tcl')
    tcl = re.sub(r'set part "[^"]*"', f'set part "{sel["part"]}"', open(tcl_path).read())
    open(tcl_path, 'w').write(tcl)
    json.dump(meta, open(meta_path, 'w'), indent=1)
    say(f"start {job['job']}: {sel['tool']} {sel['tool_version']} part {sel['part']}")

    csim = run_tool(sel, prj, 'csim', 1, 0, 1, os.path.join(logs, 'csim.log'), 3600)
    compare = subprocess.run(['python3', os.path.join(HERE, 'compare_csim.py'), prj], stdout=subprocess.PIPE, universal_newlines=True)
    meta['csim'] = csim
    try:
        meta['csim_compare'] = json.loads(compare.stdout)
    except ValueError:
        meta['csim_compare'] = {'ok': False, 'error': compare.stdout[:300]}
    if compare.returncode != 0:
        meta['status'] = 'csim_mismatch'
        say(f"{job['job']}: C-sim does not match the expected 100 outputs, synthesis skipped: {meta['csim_compare']}. See {logs}/csim.log")
    else:
        synth = run_tool(sel, prj, 'csynth', 0, 1, 0, os.path.join(logs, 'csynth.log'), TIMEOUT_H * 3600)
        meta['csynth'] = synth
        report = os.path.join(prj, 'myproject_prj', 'solution1', 'syn', 'report', 'myproject_csynth.xml')
        meta['status'] = 'ok' if synth['rc'] == 0 and os.path.exists(report) else ('timeout' if synth['rc'] == 124 else 'synth_failed')
    meta['end_utc'] = now()
    json.dump(meta, open(meta_path, 'w'), indent=1)
    say(f"{job['job']} {meta['status']}")
    print(collect.collect_job(work), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('jobs', nargs='*')
    ap.add_argument('--stage')
    args = ap.parse_args()
    lines = open(os.path.join(HERE, 'jobs.tsv')).read().splitlines()
    table = [dict(zip(lines[0].split('\t'), l.split('\t'))) for l in lines[1:]]
    wanted = [j for j in table if j['job'] in args.jobs or j['stage'] == args.stage]
    unknown = set(args.jobs) - {j['job'] for j in table}
    if unknown or not wanted:
        raise SystemExit(f'unknown jobs {sorted(unknown)}; choose from jobs.tsv or --stage A/B/C')
    selected_path = os.path.join(HERE, 'preflight', 'selected.tsv')
    if not os.path.exists(selected_path):
        raise SystemExit('run python3 preflight.py first')
    selected = {}
    for line in open(selected_path).read().splitlines():
        key, part, tool, version, settings = line.split('\t')
        selected[key] = dict(part=part, tool=tool, tool_version=version, settings=settings)
    for job in wanted:
        if job['part_key'] not in selected:
            say(f"{job['job']} skipped: preflight did not pass for {job['part_key']}, see preflight/preflight_report.txt")
            continue
        run_job(job, selected[job['part_key']])


main()
