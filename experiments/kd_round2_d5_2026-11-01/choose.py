# Created: 2026-11-01 | Last updated: 2026-11-01
# Pre-registered pick from the sweep table: best = rank 1 (lowest eval-block pL). Round 2 runs only if best
# beats uncorrelated MWPM (eval x MWPM < 1). Round-2 (alpha, T) = best's, or the best alpha < 1 cell's when
# best is the hard-label run (no teacher signal to pass on). Writes choice.json and ta_teacher.sh.
#   ~/tfenv/bin/python choose.py <sweep_results.json>
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(sys.argv[1]))['rows']
best = min(rows, key=lambda r: r['rank'])
soft = min((r for r in rows if r['alpha'] < 1), key=lambda r: r['rank'])
r2 = best if best['alpha'] < 1 else soft
choice = {'best_run': best['run'], 'best_alpha': best['alpha'], 'best_T': best['T'],
          'best_eval_pL': best['eval_pL'], 'best_eval_x_mwpm': best['eval_x_mwpm'],
          'best_weights': best['weights'], 'best_selected_epoch': best['selected_epoch'],
          'best_eval_errors': best['eval_errors'],
          'round2': best['eval_x_mwpm'] < 1.0,
          'teacher_run': r2['run'], 'teacher_weights': r2['weights'], 'teacher_epoch': r2['selected_epoch'],
          'teacher_eval_errors': r2['eval_errors'],
          'r2_alpha': f"{r2['alpha']:g}" if r2['alpha'] not in (0, 1) else f"{r2['alpha']:.1f}",
          'r2_T': r2['T'] if r2['T'] != '-' else '1.0'}
json.dump(choice, open(os.path.join(HERE, 'choice.json'), 'w'), indent=1)
dump_dir = os.path.dirname(r2['weights'])
with open(os.path.join(HERE, 'ta_teacher.sh'), 'w') as f:
    f.write(f"""# written by choose.py: round-2 teacher = sweep run {r2['run']}
T_NAME={r2['run']}; T_UNITS="140"; T_PARAMS=69441; T_EPOCH={r2['selected_epoch']}; T_EVAL_ERR={r2['eval_errors']}
T_SRC={r2['weights']}
T_DUMP={dump_dir}/eval_selected_per_shot.npz
T_COPY=$HERE/teachers/{r2['run']}_epoch{r2['selected_epoch']:03d}.weights.h5
G0=$HERE/teacher_check
""")
print(json.dumps(choice, indent=1))
