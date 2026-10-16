# Created: 2026-10-03 | Last updated: 2026-10-03
"""Copy hls4ml (main 2ed80fb9) to a separate folder and apply a one-line fix to the Vivado GRU
config template, for comparison runs only.

Bug on main 2ed80fb9, hls4ml/backends/vivado/passes/recurrent_templates.py (GRU, Resource):
the recurrent multiply (n_in = units) is written with reuse_factor = recurrent_reuse_factor,
but its dense kernel is chosen from mult_params2['reuse_factor'], which is the layer's
input-multiply reuse factor. When the two are snapped to different values on opposite sides of
n_in (u70, requested 70: input 72, recurrent 70), the RF > n_in kernel is generated with
RF = n_in and its assert fails in C simulation. The fix chooses the kernel from the recurrent
reuse factor that is actually written.

Usage: ~/hls4ml_main_env/bin/python patch_recurrent_kernel_choice.py <dest folder>
Then run with PYTHONPATH=<dest folder>. Results get the suffix _kernelfix.
"""
import os
import shutil
import sys

import hls4ml

dest = os.path.abspath(sys.argv[1])
src = os.path.dirname(hls4ml.__file__)
if os.path.exists(dest):
    shutil.rmtree(dest)
shutil.copytree(src, os.path.join(dest, 'hls4ml'))
f = os.path.join(dest, 'hls4ml', 'backends', 'vivado', 'passes', 'recurrent_templates.py')
s = open(f).read()
old = "            if int(mult_params2['reuse_factor']) <= int(mult_params2['n_in']):"
new = ("            # LOCAL FIX 2026-10-03: choose the kernel from the recurrent reuse factor that is written\n"
       "            if int(mult_params2['reuse']) <= int(mult_params2['n_in']):")
assert s.count(old) == 1, 'unexpected recurrent_templates.py'
open(f, 'w').write(s.replace(old, new))
print('patched', f)
