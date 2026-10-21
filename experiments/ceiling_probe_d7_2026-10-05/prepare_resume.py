# Created: 2026-10-13
# Last modified: 2026-10-13

"""Set up a new run folder to continue a probe arm stopped by pause_probe.sh.

Copies the best checkpoint files to <out>/seed<S>/ckpt and .lastepoch.weights.h5 (model and
optimizer state) to <out>/resume_from, and rebuilds the history of the finished epochs from
run.log. Prints KEY=value resume settings for gh200_run_student_scaling.sh. Standard library only.

python3 prepare_resume.py <paused run folder> <new run folder> <seed>
"""
import glob
import json
import os
import re
import shutil
import sys

src, out, seed = sys.argv[1], sys.argv[2], sys.argv[3]
sd = os.path.join(src, f'seed{seed}')
txt = open(os.path.join(sd, 'run.log'), errors='replace').read()
lines = [l for l in txt.splitlines() if 'ms/step' in l]
n = len(lines)
paused = open(os.path.join(src, 'PAUSED')).read() if os.path.exists(os.path.join(src, 'PAUSED')) else ''
m = re.search(r'paused after epoch (\d+)', paused)
if not m or int(m.group(1)) != n:
    raise SystemExit(f'{src}: {n} epoch lines in run.log, PAUSED says {paused.strip()!r}. STOP.')

tag = os.path.basename(glob.glob(os.path.join(sd, 'ckpt', '*.best.json'))[0])[:-len('.best.json')]
ck, rf = os.path.join(out, f'seed{seed}', 'ckpt'), os.path.join(out, 'resume_from')
os.makedirs(ck)
os.makedirs(rf)
for ext in ('.best.keras', '.best.weights.h5', '.best.json'):
    shutil.copy2(os.path.join(sd, 'ckpt', tag + ext), ck)
shutil.copy2(os.path.join(sd, 'ckpt', tag + '.lastepoch.weights.h5'), rf)

h = {}
for l in lines:
    for k, v in re.findall(r' - ([a-z_]+): ([-0-9.e+]+)', l):
        h.setdefault(k, []).append(float(v))
if not h or not all(len(v) == n for v in h.values()):
    raise SystemExit('history rebuild: unequal lengths ' + str({k: len(v) for k, v in h.items()}))
json.dump(h, open(os.path.join(rf, tag + '.history.json'), 'w'))

print(f'RESUME_WEIGHTS={os.path.join(rf, tag + ".lastepoch.weights.h5")}')
print(f'INITIAL_EPOCH={n}')
print(f'RESUME_BEST_JSON={os.path.join(ck, tag + ".best.json")}')
print(f'RESUME_HISTORY={os.path.join(rf, tag + ".history.json")}')
