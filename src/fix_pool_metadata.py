#!/usr/bin/env python3
# Created: 2026-10-06
# Last modified: 2026-10-06

"""Fix stale fields in the p = 0.004 FORMAL pool fingerprints and their pool_provenance.json copies.

Written before the validation block was split:
  partition_val [15M, 17M)  -> validation [15.0M, 15.2M) and evaluation [15.2M, 17M)
  mwpm_resid_errors_test    -> mwpm_resid_errors_15M_17M (decoded on [15M, 17M), not the sealed block)
Old values are kept under "as_written_before_2026-10-06". Running twice changes nothing.

python fix_pool_metadata.py <json files...>
"""
import json
import sys

STALE_VAL = [15000000, 17000000]


def fix(path):
    m = json.load(open(path))
    if m.get('partition_val') != STALE_VAL and 'mwpm_resid_errors_test' not in m:
        return None
    old = {k: m[k] for k in ('partition_val', 'n_val', 'n_test', 'mwpm_resid_errors_test')
           if k in m}
    notes = []
    if m.get('partition_val') == STALE_VAL:
        m['partition_val'] = [15000000, 15200000]
        m['partition_eval'] = [15200000, 17000000]
        m['n_val'] = 200000
        m['n_eval'] = 1800000
        notes.append('partition_val [15M, 17M] (written before the split) -> validation '
                     '[15.0M, 15.2M) and evaluation [15.2M, 17M), as configs/partitions.yaml')
    if 'mwpm_resid_errors_test' in m:
        m['mwpm_resid_errors_15M_17M'] = m.pop('mwpm_resid_errors_test')
        m['mwpm_block'] = [15000000, 17000000]
        notes.append('mwpm_resid_errors_test -> mwpm_resid_errors_15M_17M: gen_pool_r5.py '
                     'decoded MWPM on [15M, 17M) (validation + evaluation), not the sealed '
                     'block; mwpm_p_L is over the same 2M shots')
    if 'n_test' in m:
        m['n_sealed'] = m.pop('n_test')
        notes.append('n_test -> n_sealed (the size of partition_test_SEALED)')
    m['as_written_before_2026-10-06'] = old
    m['corrections_2026-10-06'] = notes
    json.dump(m, open(path, 'w'), indent=2)
    return notes


if __name__ == '__main__':
    for p in sys.argv[1:]:
        n = fix(p)
        print(('fixed  ' if n else 'as is  ') + p)
