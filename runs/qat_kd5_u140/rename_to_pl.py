# Created: 2026-10-28 | Last updated: 2026-10-28
# One-off rename of the logical error rate to pL in this study's code and result files:
# keys and columns ler* -> pL*, *p_L* -> *pL*, text LER -> pL. Values are unchanged; raw logs/ are left as written.
#   python3 rename_to_pl.py
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PATTERNS = ['*.py', '*.md', 'results/*.json', 'results/*.csv', 'qat_runs/*/result.json', 'qat_runs/*/history.jsonl',
            'specs/*.json', 'eaf_bundle/*.md']


def rename(text):
    text = re.sub(r'(?<![A-Za-z0-9])ler(?![A-Za-z])', 'pL', text)
    text = re.sub(r'(?<![A-Za-z0-9])LERs(?![A-Za-z])', 'pL values', text)
    text = re.sub(r'(?<![A-Za-z0-9])LER(?![A-Za-z])', 'pL', text)
    return re.sub(r'(?<![A-Za-z0-9])p_L(?![A-Za-z0-9])', 'pL', text)


changed = []
for pat in PATTERNS:
    for p in sorted(glob.glob(os.path.join(HERE, pat))):
        if os.path.basename(p) == 'rename_to_pl.py':
            continue
        old = open(p).read()
        new = rename(old)
        if new != old:
            open(p, 'w').write(new)
            changed.append(os.path.relpath(p, HERE))
print(len(changed), 'files changed')
print('\n'.join(changed))
