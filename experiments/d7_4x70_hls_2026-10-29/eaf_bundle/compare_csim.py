# Created: 2026-10-29 | Last updated: 2026-10-29
# Vivado HLS C-sim output (tb_data/csim_results.log) vs the expected output written on grace1 (hls4ml C-sim of
# the same project). Exit 0 if all 100 decisions match and every logit is within 1e-4.
#   python3 compare_csim.py projects/<project>
import json
import os
import sys

prj = sys.argv[1]
read = lambda f: [float(l.split()[0]) for l in open(os.path.join(prj, 'tb_data', f)) if l.strip()]
try:
    got, exp = read('csim_results.log'), read('tb_output_predictions.dat')
except OSError as e:
    print(json.dumps({'project': prj, 'ok': False, 'error': str(e)}))
    sys.exit(1)
n = min(len(got), len(exp))
diff = max(abs(a - b) for a, b in zip(got, exp)) if n else float('nan')
same = sum((a > 0) == (b > 0) for a, b in zip(got, exp))
ok = n == len(exp) == 100 and same == n and diff <= 1e-4
print(json.dumps({'project': os.path.basename(prj.rstrip('/')), 'ok': ok, 'n': n, 'decisions_equal': same,
                  'max_abs_diff': diff}))
sys.exit(0 if ok else 1)
