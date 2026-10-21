# Created: 2026-10-03 | Last updated: 2026-10-03

"""Copy hls4ml main 2ed80fb9 to a separate folder and apply a one-line fix to the Vivado GRU template.

On 2ed80fb9 the recurrent multiply is written with recurrent_reuse_factor but its dense kernel is
chosen from the input-multiply reuse factor. When the two snap to opposite sides of n_in (u70:
input 72, recurrent 70) the RF > n_in kernel fails its assert in C-sim. The fix chooses the
kernel from the recurrent reuse factor. Results get the suffix _kernelfix.

~/hls4ml_main_env/bin/python patch_recurrent_kernel_choice.py <dest folder>
then run with PYTHONPATH=<dest folder>
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
