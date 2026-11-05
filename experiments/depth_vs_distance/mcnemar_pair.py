# Created: 2026-11-05 | Last updated: 2026-11-05
# paired McNemar on the eval block: dvd_d9_L03 (peak 1e-3) vs f3280 (peak 5e-4), both epoch 160
import json, sys
import numpy as np
from scipy.stats import binomtest, chi2
a = np.load(sys.argv[1]); b = np.load(sys.argv[2])
assert np.array_equal(a["shot_idx"], b["shot_idx"]) and np.array_equal(a["truth"], b["truth"])
ca, cb = a["student_correct"], b["student_correct"]
n_a_only = int((ca & ~cb).sum()); n_b_only = int((~ca & cb).sum())
n_disc = n_a_only + n_b_only
stat = (abs(n_a_only - n_b_only) - 1) ** 2 / n_disc
out = dict(n=int(ca.size), errors_dvd_d9_L03=int((~ca).sum()), errors_f3280=int((~cb).sum()),
           pL_dvd_d9_L03=float((~ca).mean()), pL_f3280=float((~cb).mean()),
           dvd_only_right=n_a_only, f3280_only_right=n_b_only, n_disc=n_disc,
           chi2_cc=stat, p_chi2=float(chi2.sf(stat, 1)),
           p_exact=float(binomtest(n_a_only, n_disc, 0.5).pvalue))
print(json.dumps(out, indent=1))
json.dump(out, open("mcnemar_dvd_d9_L03_vs_f3280.json", "w"), indent=1)
