#!/usr/bin/env bash
# Created: 2026-10-18
# Last modified: 2026-10-18
# Timing runs, one at a time on an idle GPU. Projection uses epoch 2 (epoch 1 includes tracing).
#   bash smoke.sh [run ...]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HOSTPY=${HOSTPY:-$HOME/tfenv/bin/python}
RUNS=("$@"); [ ${#RUNS[@]} -gt 0 ] || RUNS=(kd5_u140 hl5_u140 kd7_2x140 hl7_2x140 kd9_3x140)
mkdir -p "$HERE/smoke"
RES="$HERE/smoke/resources.tsv"
( while true; do
    t=$(date -u +%FT%TZ)
    podman stats --no-stream --format '{{.Name}}\t{{.MemUsage}}' 2>/dev/null | sed "s/^/$t\thost\t/"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | sed "s/^/$t\tgpu\t/"
    sleep 15
  done ) >> "$RES" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
for r in "${RUNS[@]}"; do
  [ -n "$(podman ps -q)" ] && { echo "another container is running; timing runs need the GPU alone. STOP."; exit 1; }
  SMOKE=1 bash "$HERE/run_kd.sh" "$r" || echo "smoke $r FAILED"
done
kill $SAMPLER 2>/dev/null

"$HOSTPY" - "$HERE/smoke" "$RES" "${RUNS[@]}" <<'PY'
import glob, os, re, sys
smoke, res, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
full_epochs = {'5': 80, '7': 80, '9': 160}
def gib(s):
    m = re.match(r'([\d.]+)\s*([KMG]i?B)', s.strip())
    v, u = float(m.group(1)), m.group(2)[0]
    return v / {'K': 2**20, 'M': 2**10, 'G': 1}[u]
peak_host = {}
for line in open(res):
    t, kind, rest = line.rstrip('\n').split('\t', 2)
    if kind == 'host':
        name, mem = rest.split('\t')
        peak_host[name] = max(peak_host.get(name, 0), gib(mem.split('/')[0]))
gpu = [float(l.rstrip().split(',')[-1].split()[0]) for l in open(res) if '\tgpu\t' in l]
rows = ['run\tepoch1_s\tepoch2_s\tprojected_100M_epoch_s\tprojected_train_h\tpeak_host_GiB\tpeak_gpu_MiB_any\tdone']
for r in runs:
    out = sorted(glob.glob(os.path.join(smoke, f'{r}_*')))
    if not out:
        rows.append(f'{r}\t\t\t\t\t\t\tmissing')
        continue
    log = open(os.path.join(out[-1], 'seed0', 'run.log')).read()
    secs = [int(s) for s in re.findall(r'^1000/1000 - (\d+)s', log, re.M)]
    d = r[2]
    e2 = secs[1] if len(secs) > 1 else None
    proj = e2 * 10 if e2 else None
    rows.append('\t'.join(str(v) for v in (
        r, secs[0] if secs else '', e2 or '', proj or '',
        round(proj * full_epochs[d] / 3600, 2) if proj else '',
        round(peak_host.get(f'kd_{r}', 0), 1), max(gpu) if gpu else '',
        int(os.path.exists(os.path.join(out[-1], 'DONE'))))))
open(os.path.join(smoke, 'timing.tsv'), 'w').write('\n'.join(rows) + '\n')
print('\n'.join(rows))
PY
