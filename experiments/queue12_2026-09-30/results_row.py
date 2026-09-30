# Created: 2026-09-30
# Last modified: 2026-09-30
"""One tab-separated results line for a finished run, in the column order of the
experiment-plan table (plus clipping columns), printed to stdout.

    python results_row.py <run folder>

Columns: Done, width, params, d, r, LR schedule, peak LR, training shots, seed, p, batch,
max epoch, best ckpt epoch, MWPM p_L, GRU p_L, GRU/MWPM, val loss at best ckpt,
reset_after, checkpoint rule, global_clipnorm, fraction of updates clipped, warmup, run.
"""
import json
import os
import re
import sys

run = sys.argv[1]
sd = os.path.join(run, 'seed0')
s = json.load(open(os.path.join(sd, 'summary.json')))
log = open(os.path.join(sd, 'run.log')).read()
cmd = open(os.path.join(run, 'command.txt')).read()
env = dict(re.findall(r'\b([A-Z_]+)=(\S+)', cmd))
tot = re.findall(r'total ([\d,]+)', log)
ntr = int(tot[-1].replace(',', '')) if tot else int(env['NTRAIN'])
shots = f'{ntr // 1_000_000}M'
row = ['Yes', f"u{env['UNITS']}", s['params'], env['D'], env['ROUNDS'], env['LR_SCHEDULE'],
       env['LR'], shots, env['SEEDS'], env['P'], env['BATCH'], env['EPOCHS'], s['best_epoch'],
       s['mwpm_p_L'], s['p_L'], round(s['p_L'] / s['mwpm_p_L'], 3), s['best_val_loss'],
       'TRUE' if env.get('RESET_AFTER') == '1' else 'FALSE', 'min val p_L',
       env.get('GLOBAL_CLIPNORM', ''), s.get('clip_frac_cum', ''), 'no',
       os.path.basename(run)]
print('\t'.join(str(x) for x in row))
