# Created: 2026-10-28 | Last updated: 2026-10-28
# hls4ml C-simulation helpers (host CPU env: ~/hls4ml_main_env python, PYTHONPATH=hls4ml main worktree)
import multiprocessing as mp
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np

_HM = {}


def _work(task):
    a, xb = task
    return a, np.asarray(_HM['m'].predict(np.ascontiguousarray(xb)), float).reshape(-1)


def csim_predict(hm, x, workers=32, chunk=2000):
    # hls_model.predict in parallel chunks (fork shares the compiled library)
    _HM['m'] = hm
    x = np.ascontiguousarray(x.astype(np.float32))
    tasks = [(a, x[a:a + chunk]) for a in range(0, len(x), chunk)]
    y = np.empty(len(x))
    with mp.get_context('fork').Pool(min(workers, len(tasks))) as pool:
        for a, yy in pool.imap_unordered(_work, tasks):
            y[a:a + len(yy)] = yy
    return y


def weights_txt(prj, name):
    import re
    txt = open(os.path.join(prj, 'firmware', 'weights', f'{name}.h')).read()
    body = txt[txt.index('{') + 1:txt.rindex('}')]
    return np.array([float(v) for v in re.split(r'[,\s]+', body.strip()) if v])


def generated_types(prj):
    # typedefs and the GRU config of the generated project
    import re
    defines = open(os.path.join(prj, 'firmware', 'defines.h')).read()
    params = open(os.path.join(prj, 'firmware', 'parameters.h')).read()
    return {'typedefs': dict((b, a) for a, b in re.findall(r'typedef (.+?) (\w+);', defines)),
            'reset_after': re.findall(r'static const bool reset_after = (\w+);', params),
            'table_t': re.findall(r'typedef (.+?) table_t;', params),
            'accum_t': re.findall(r'typedef (.+?) accum_t;', params),
            'n_unit_par': re.findall(r'n_unit_par = (\d+);', params),
            'n_cycles': re.findall(r'n_cycles = (\d+);', params),
            'reuse_factor': re.findall(r'reuse_factor = (\d+);', params)}
