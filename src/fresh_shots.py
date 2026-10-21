# Created: 2026-10-09
# Last modified: 2026-10-18 (check_seeds: base must be registered in partitions.yaml)
"""Fresh training shots: a new Stim sample for every batch.

Same circuit as the pools (circuit_generators.build_circuit). Chunk c of epoch e uses seed
seed_base + e * SEED_EPOCH_STRIDE + c, so a resumed run draws the same shots. Validation and
scored blocks come from the FORMAL pool.
"""
import multiprocessing as mp

import numpy as np

SEED_BASE = 10 ** 12
SEED_EPOCH_STRIDE = 10 ** 6          # chunks per epoch must stay below this

_W = {}                              # per-worker state, set by _init


def _init(d, p, rounds, index_map):
    import stim  # noqa: F401  (imported once per worker)
    from circuit_generators import build_circuit
    _W['circ'] = build_circuit(d, p, rounds)
    _W['index_map'] = index_map


def _scatter(det, index_map):
    # [n, n_det] -> [n, n_timesteps, n_positions] uint8, same layout as StudentModels.to_sequence.
    n_t, n_pos = index_map.shape
    out = np.zeros((det.shape[0], n_t, n_pos), dtype=np.uint8)
    for t in range(n_t):
        cols = index_map[t]
        present = cols >= 0
        out[:, t, present] = det[:, cols[present]]
    return out


def _sample(task):
    # (seed, n) -> (x uint8 [n, T, P], flips float32 [n]).
    seed, n = task
    sampler = _W['circ'].compile_detector_sampler(seed=int(seed))
    det, obs = sampler.sample(int(n), separate_observables=True)
    return _scatter(det.astype(np.uint8), _W['index_map']), obs.reshape(n, -1)[:, 0].astype(np.float32)


def chunk_seed(epoch, chunk, seed_base=SEED_BASE):
    return seed_base + epoch * SEED_EPOCH_STRIDE + chunk


def seed_range(epochs, chunks_per_epoch, seed_base=SEED_BASE):
    return seed_base, seed_base + (epochs - 1) * SEED_EPOCH_STRIDE + chunks_per_epoch


def check_seeds(epochs, chunks_per_epoch, seed_base=SEED_BASE):
    # Refuse seeds that overlap configs/partitions.yaml or an unregistered seed base.
    import partitions as P
    if chunks_per_epoch >= SEED_EPOCH_STRIDE:
        raise ValueError(f'{chunks_per_epoch} chunks per epoch >= stride {SEED_EPOCH_STRIDE}')
    lo, hi = seed_range(epochs, chunks_per_epoch, seed_base)
    clash = sorted(s for s in P.used_seeds() if lo <= s < hi)
    if clash:
        raise ValueError(f'fresh-shot seeds [{lo}, {hi}) include pool seeds {clash}')
    entry = [(r0, r1, n) for r0, r1, n in P.fresh_seed_ranges() if r0 == seed_base]
    if not entry:
        raise ValueError(f'fresh seed base {seed_base} is not in configs/partitions.yaml '
                         f'fresh_seed_ranges; register it there first')
    if epochs > entry[0][2]:
        raise ValueError(f'{epochs} epochs from base {seed_base} exceed its registered '
                         f'max_epochs {entry[0][2]}')
    return lo, hi


class FreshShots:
    # y[:, 0] is the logical flip, y[:, 1] the teacher-logit column (zero here).

    def __init__(self, d, p, rounds, shots_per_epoch, batch_size, chunk=100_000,
                 workers=16, seed_base=SEED_BASE, in_flight=None):
        from StudentModels import detector_sequence_layout
        chunk = min(chunk, shots_per_epoch)          # small test runs: one chunk per epoch
        if chunk % batch_size or shots_per_epoch % chunk:
            raise ValueError(f'need batch_size | chunk | shots_per_epoch; got {batch_size}, '
                             f'{chunk}, {shots_per_epoch}')
        n_t, n_pos, index_map = detector_sequence_layout(d, rounds, p)
        self.shape = (n_t, n_pos)
        self.d, self.p, self.rounds = d, p, rounds
        self.shots_per_epoch, self.batch_size, self.chunk = shots_per_epoch, batch_size, chunk
        self.chunks_per_epoch = shots_per_epoch // chunk
        self.seed_base = seed_base
        self.in_flight = in_flight or 2 * workers     # chunks requested ahead, at most
        self.pool = mp.get_context('spawn').Pool(workers, initializer=_init,
                                                 initargs=(d, p, rounds, index_map))

    def batches(self, epoch):
        import collections
        tasks = iter([(chunk_seed(epoch, c, self.seed_base), self.chunk)
                      for c in range(self.chunks_per_epoch)])
        pending = collections.deque()
        for t in tasks:                          # fill the window
            pending.append(self.pool.apply_async(_sample, (t,)))
            if len(pending) >= self.in_flight:
                break
        while pending:
            x, f = pending.popleft().get()
            t = next(tasks, None)
            if t is not None:                    # one out, one in: the window stays bounded
                pending.append(self.pool.apply_async(_sample, (t,)))
            y = np.stack([f, np.zeros_like(f)], axis=1)
            for a in range(0, self.chunk, self.batch_size):
                yield x[a:a + self.batch_size], y[a:a + self.batch_size]

    def dataset(self, tf, first_epoch=0, to_gpu=True):
        #first_epoch: the --initial-epoch of a resumed run.
        state = {'epoch': first_epoch}
        T, P = self.shape

        def gen():
            e = state['epoch']
            state['epoch'] += 1
            for x, y in self.batches(e):
                yield x, y

        with tf.device('/CPU:0'):
            ds = tf.data.Dataset.from_generator(
                gen, output_signature=(tf.TensorSpec((None, T, P), tf.uint8),
                                       tf.TensorSpec((None, 2), tf.float32)))
            ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y))
            ds = ds.prefetch(8)
        if to_gpu and tf.config.list_logical_devices('GPU'):
            ds = ds.apply(tf.data.experimental.prefetch_to_device('/GPU:0', buffer_size=2))
        return ds

    def close(self):
        self.pool.close()
        self.pool.join()
