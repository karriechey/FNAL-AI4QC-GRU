# Created: 2026-10-21
# Last modified: 2026-10-21

"""LR stability scan (prereg/depth_scaling_d7_2026-10.md): kill rule while training, then
diverged (killed or NaN) / unstable (train loss > 1.10 x running min for 3+ epochs) / stable.

python classify_lr_scan.py --kill-check <run folder>   exit 3 and KILLED/DIVERGED when the rule fires
python classify_lr_scan.py                              lr_scan/classification.{csv,md}, lr_scan/sweep_lr.json
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'mwpm_fixes_d9_2026-10'))
from driver import epoch_val_ler, streak_reason

N_DET = 336                  # d=7, r=7: 48 detectors x 7 rounds
UNSTABLE_FACTOR = 1.10
UNSTABLE_EPOCHS = 3
SCAN_EPOCHS = 20


def run_log(out):
    f = os.path.join(out, 'seed0', 'run.log')
    return open(f, errors='replace').read() if os.path.exists(f) else ''


def train_losses(out):
    # per-epoch training loss: history.json when the run finished, the run log otherwise
    h = glob.glob(os.path.join(out, 'seed0', 'runs', '*.history.json'))
    if h:
        return json.load(open(h[0]))['loss']
    return [float(v) for v in re.findall(r'^\d+/\d+ - \d+s - loss: (\S+)', run_log(out), re.M)]


def kill_check(out):
    txt = run_log(out)
    if re.search(r'loss: nan|loss=nan', txt):
        ep = len(re.findall(r'^\d+/\d+ - \d+s - loss', txt, re.M))
        open(os.path.join(out, 'DIVERGED'), 'w').write(f'training loss nan at epoch {ep}\n')
        print(f'DIVERGED epoch {ep}')
        return 3
    r = streak_reason(epoch_val_ler(out))
    if r is not None:
        open(os.path.join(out, 'KILLED'), 'w').write(f'Killed @ ep {r[0]}: {r[1]}\n')
        print(f'KILLED epoch {r[0]}: {r[1]}')
        return 3
    return 0


def longest_rise(losses):
    # (longest run of epochs above UNSTABLE_FACTOR x the running minimum of earlier epochs, max ratio)
    best, run, longest, worst = math.inf, 0, 0, 0.0
    for v in losses:
        if best < math.inf:
            worst = max(worst, v / best)
        run = run + 1 if v > UNSTABLE_FACTOR * best else 0
        longest = max(longest, run)
        best = min(best, v)
    return longest, worst


def classify(out):
    killed = os.path.exists(os.path.join(out, 'KILLED'))
    nan = os.path.exists(os.path.join(out, 'DIVERGED'))
    losses = train_losses(out)
    longest, worst = longest_rise(losses)
    if killed or nan or any(math.isnan(v) for v in losses):
        cls = 'diverged'
    elif longest >= UNSTABLE_EPOCHS:
        cls = 'unstable'
    else:
        cls = 'stable'
    note = ''
    for f in ('KILLED', 'DIVERGED'):
        if os.path.exists(os.path.join(out, f)):
            note = open(os.path.join(out, f)).read().strip()
    gn = re.findall(r'global grad norm median (\S+); clipped [\d,]+/[\d,]+ updates \((\S+)\)', run_log(out))
    m = re.search(r'Killed @ ep (\d+)', note)
    return dict(cls=cls, epochs=len(losses), final_loss=losses[-1] if losses else None,
                min_loss=min(losses) if losses else None, longest_rise=longest, max_rise=round(worst, 3),
                killed_epoch=int(m.group(1)) if m else (len(losses) if nan else ''),
                grad_norm_median_max=max((float(g) for g, _ in gn), default=None),
                clip_frac_max=max((float(c) for _, c in gn), default=None),
                complete=os.path.exists(os.path.join(out, 'DONE')), note=note)


def scan_runs():
    runs = {}
    for out in sorted(glob.glob(os.path.join(HERE, 'lr_scan', 'lrscan_d7_L*_lr*_2*'))):
        name = re.sub(r'_\d{8}T\d{6}Z$', '', os.path.basename(out))
        runs[name] = out           # the newest folder of a name
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kill-check')
    a = ap.parse_args()
    if a.kill_check:
        sys.exit(kill_check(a.kill_check))

    rows = []
    for name, out in scan_runs().items():
        m = re.match(r'lrscan_d7_L(\d+)_lr(\S+)$', name)
        L, lr = int(m.group(1)), m.group(2)
        c = classify(out)
        if not c['complete'] and c['cls'] != 'diverged':
            raise SystemExit(f'{name}: neither finished nor killed ({c["epochs"]} epochs)')
        if c['complete'] and c['epochs'] != SCAN_EPOCHS:
            raise SystemExit(f'{name}: {c["epochs"]} epochs, expected {SCAN_EPOCHS}')
        rows.append(dict(run=name, L=L, peak_lr=lr, eta_ndet=round(float(lr) * N_DET, 2),
                         status=c['cls'], killed_epoch=c['killed_epoch'], epochs=c['epochs'],
                         final_train_loss=c['final_loss'], min_train_loss=c['min_loss'],
                         longest_rise_epochs=c['longest_rise'], max_loss_over_running_min=c['max_rise'],
                         grad_norm_median_max=c['grad_norm_median_max'], clip_frac_max=c['clip_frac_max'],
                         folder=os.path.basename(out), note=c['note']))
    rows.sort(key=lambda r: (r['L'], float(r['peak_lr'])))
    base = os.path.join(HERE, 'lr_scan')
    with open(os.path.join(base, 'classification.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = ['| run | L | peak LR | eta x N_det | status | epochs | final train loss | longest rise (epochs) '
          '| max loss / running min | max epoch grad norm median | max epoch clip fraction | note |',
          '|' + '---|' * 12]
    for r in rows:
        md.append(f"| {r['run']} | {r['L']} | {r['peak_lr']} | {r['eta_ndet']} | {r['status']} | {r['epochs']} "
                  f"| {r['final_train_loss']:.5f} | {r['longest_rise_epochs']} | {r['max_loss_over_running_min']} "
                  f"| {r['grad_norm_median_max']:.4g} | {r['clip_frac_max']:.4f} | {r['note']} |")
    open(os.path.join(base, 'classification.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))

    # sweep LR rule
    st = {(r['L'], r['peak_lr']): r['status'] for r in rows}
    s1, s05 = st.get((10, '1e-3')), st.get((10, '5e-4'))
    if s1 == 'stable':
        rule = dict(sweep_lr='1e-3', reason='L=10 at 1e-3 stable')
    elif s1 is None:
        rule = dict(sweep_lr=None, reason='L=10 at 1e-3 not run yet')
    elif s05 is None:
        rule = dict(sweep_lr=None, need='lrscan_d7_L10_lr5e-4', reason=f'L=10 at 1e-3 {s1}; 5e-4 run needed')
    elif s05 == 'stable':
        rule = dict(sweep_lr='5e-4', reason=f'L=10 at 1e-3 {s1}, at 5e-4 stable')
    else:
        rule = dict(sweep_lr=None, stop=True, reason=f'L=10 at 1e-3 {s1}, at 5e-4 {s05}: neither stable')
    json.dump(rule, open(os.path.join(base, 'sweep_lr.json'), 'w'), indent=1)
    print(json.dumps(rule))


if __name__ == '__main__':
    main()
