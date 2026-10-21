#!/usr/bin/env python3
# from FNAL-QCDecoding-FPGA/generate_pools.py @ d5ee86f
# Last modified: 2026-09-30 (sample_pool_compact: same shots as sample_pool, detection events bit-packed, no measurements).
# Earlier 2026-09-25 (build_circuit delegates to circuit_generators.build_circuit)

"""Generate the 8 core pools for the RCNN-vs-MWPM threshold study, and the MWPM baseline per (d, p).

rounds = d, all four Stim noise channels = p, same circuit for sampling and decoding.
Training prefixes come from the front; the test tail is the last N_test shots.
Writes data_d{d}_p{p:.3f}_r{d}.npz and mwpm_baseline.csv to DATA_ROOT.
"""
import csv
import os
import time
import numpy as np
from circuit_generators import build_circuit as _build_circuit  # the single noise-model definition

# --- CONFIG -----------------------------------------------------------------------------
DATA_ROOT = os.path.expanduser("~/rcnn_threshold/pools")
GEN_SEED = 12345          # fixed; recorded per pool for provenance
CHUNK = 1_000_000         # sample in chunks to cap peak RAM on the 10M pools
OBS = "ZL"                # rotated_memory_z single logical observable

# (d, p, N_total, N_test); N_total = train capacity + N_test
# train capacity: 10M for p<=0.005, 8M for p=0.007, 5M for p=0.010
POOLS = [
    # d=3, rounds=3
    (3, 0.004, 10_050_000, 50_000),
    (3, 0.005, 10_030_000, 30_000),
    (3, 0.007,  8_020_000, 20_000),
    (3, 0.010,  5_010_000, 10_000),
    # d=5, rounds=5
    (5, 0.004, 10_070_000, 70_000),
    (5, 0.005, 10_040_000, 40_000),
    (5, 0.007,  8_020_000, 20_000),
    (5, 0.010,  5_010_000, 10_000),
]
# ----------------------------------------------------------------------------------------

def build_circuit(d, p):
    return _build_circuit(d, p, rounds=d)


def sample_pool(circ, n_total, seed):
    # chunked so peak RAM is one chunk plus the output arrays
    m_sampler = circ.compile_sampler(seed=seed)
    converter = circ.compile_m2d_converter()
    # widths from a 2-shot probe so we can preallocate
    probe = m_sampler.sample(2, bit_packed=False)
    det0, obs0 = converter.convert(measurements=probe, separate_observables=True, bit_packed=False)
    n_meas, n_det, n_obs = probe.shape[1], det0.shape[1], obs0.reshape(2, -1).shape[1]
    measurements = np.empty((n_total, n_meas), dtype=np.int8)
    det_evts = np.empty((n_total, n_det), dtype=np.int8)
    flips = np.empty((n_total, n_obs), dtype=np.int8)
    # fresh sampler so the 2-shot probe doesn't shift the stream
    m_sampler = circ.compile_sampler(seed=seed)
    done = 0
    while done < n_total:
        k = min(CHUNK, n_total - done)
        meas = m_sampler.sample(k, bit_packed=False)
        det, obs = converter.convert(measurements=meas, separate_observables=True, bit_packed=False)
        measurements[done:done + k] = meas.astype(np.int8)
        det_evts[done:done + k] = det.astype(np.int8)
        flips[done:done + k] = obs.astype(np.int8).reshape(k, -1)
        done += k
    return measurements, det_evts, flips


def sample_pool_compact(circ, n_total, seed):
    # Same shots as sample_pool, kept as bit-packed detection events only (~1/16 the size).
    # Returns (det_evts_packed uint8 [n, ceil(n_det/8)], n_det, flips int8 [n, n_obs]).
    # Unpack with np.unpackbits(det_evts_packed, axis=1, count=n_det).
    m_sampler = circ.compile_sampler(seed=seed)
    converter = circ.compile_m2d_converter()
    probe = m_sampler.sample(2, bit_packed=False)
    det0, obs0 = converter.convert(measurements=probe, separate_observables=True, bit_packed=False)
    n_det, n_obs = det0.shape[1], obs0.reshape(2, -1).shape[1]
    det_packed = np.empty((n_total, (n_det + 7) // 8), dtype=np.uint8)
    flips = np.empty((n_total, n_obs), dtype=np.int8)
    m_sampler = circ.compile_sampler(seed=seed)     # fresh, as in sample_pool
    done = 0
    while done < n_total:
        k = min(CHUNK, n_total - done)
        meas = m_sampler.sample(k, bit_packed=False)
        det, obs = converter.convert(measurements=meas, separate_observables=True, bit_packed=False)
        det_packed[done:done + k] = np.packbits(det.astype(np.uint8), axis=1)
        flips[done:done + k] = obs.astype(np.int8).reshape(k, -1)
        done += k
    return det_packed, n_det, flips


def main():
    import pymatching  # imported here so the script can be read without the dep present
    os.makedirs(DATA_ROOT, exist_ok=True)
    baseline_csv = os.path.join(DATA_ROOT, "mwpm_baseline.csv")
    fields = ["d", "p", "rounds", "n_total", "n_test", "train_capacity",
              "mwpm_p_L", "mwpm_resid_errors_test", "base_rate_test",
              "base_rate_pool", "gen_seed"]
    new = not os.path.exists(baseline_csv)
    f = open(baseline_csv, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()

    print(f"[gen] DATA_ROOT = {DATA_ROOT}")
    for d, p, n_total, n_test in POOLS:
        t0 = time.time()
        fn = os.path.join(DATA_ROOT, f"data_d{d}_p{p:.3f}_r{d}.npz")
        train_cap = n_total - n_test
        # top of the ladder must not reach the test tail
        assert train_cap == n_total - n_test, "train/test overlap"
        if os.path.exists(fn):
            print(f"[gen] skip (exists) {fn}")
            continue

        circ = build_circuit(d, p)
        measurements, det_evts, flips = sample_pool(circ, n_total, GEN_SEED)

        # MWPM baseline on the fixed tail
        dem = circ.detector_error_model(decompose_errors=True)
        pym = pymatching.Matching.from_detector_error_model(dem)
        te = slice(n_total - n_test, n_total)
        pred = pym.decode_batch(det_evts[te], bit_packed_predictions=False,
                                bit_packed_shots=False).astype(np.int8).reshape(-1, 1)
        truth = flips[te].reshape(-1, 1)
        resid = int((pred != truth).sum())
        mwpm_pL = resid / n_test
        base_test = float(flips[te].mean())     # all-zero predictor's error on the tail
        base_pool = float(flips.mean())          # label positive rate across the pool

        np.savez(fn, measurements=measurements, det_evts=det_evts, flips=flips)
        size_gb = os.path.getsize(fn) / 1024**3
        w.writerow(dict(d=d, p=p, rounds=d, n_total=n_total, n_test=n_test,
                        train_capacity=train_cap, mwpm_p_L=round(mwpm_pL, 6),
                        mwpm_resid_errors_test=resid, base_rate_test=round(base_test, 5),
                        base_rate_pool=round(base_pool, 5), gen_seed=GEN_SEED))
        f.flush()
        # flag a tail with too few MWPM errors
        starved = " <-- FLAG: <300 residual errors, tail too small" if resid < 300 else ""
        print(f"[gen] d={d} p={p:.3f} r={d}  N={n_total:,} (train_cap {train_cap:,}, "
              f"tail {n_test:,})  MWPM p_L={mwpm_pL:.5f} ({resid} resid errs){starved}  "
              f"base_rate(tail)={base_test:.3f}  {size_gb:.2f} GB  {time.time()-t0:.0f}s")
    f.close()
    print(f"[gen] baseline -> {baseline_csv}")
    print("[gen] DONE. Next: smoke-test ONE training job before any fan-out.")


if __name__ == "__main__":
    main()
