# Created: 2026-10-09
# Last modified: 2026-10-18 (check_seeds: base must be registered in partitions.yaml)
"""Fresh training shots: a new Stim sample for every training batch, so no shot is seen twice.

Why: at d9 the stacked GRUs fit their 100M training shots far better than new shots
(9-S, 2 x u280: 0.78x MWPM on a slice of its own training shots, 1.62x on the evaluation
block). With a new sample every epoch the model cannot memorise shots, and the amount of
training data is limited only by the number of epochs.

The shots come from the same circuit as every pool (circuit_generators.build_circuit:
rotated memory-Z, rounds = d, all four noise channels = p). They are sampled with Stim's
detector sampler, which gives the detection events and the logical flip of each shot
directly; the pools went through measurements and the measurement-to-detector converter,
which gives the same distribution (tests/fresh_shots_test.py compares the rates).

Seeds. Shots are sampled in chunks of `chunk` shots; chunk c of epoch e uses the Stim seed
    seed_base + e * SEED_EPOCH_STRIDE + c
so every chunk of every epoch is a separate stream, and a run restarted at epoch e draws
the same shots it would have drawn uninterrupted. seed_base defaults to 10**12, far above
every pool seed in configs/partitions.yaml; check_seeds() refuses a run if any pool or
held-out seed falls inside the range this run uses. Validation, evaluation and sealed
blocks are still read from the FORMAL pool; fresh shots replace only the training array.

Stim's output for a seed is fixed for one Stim version on one machine architecture (the
pools are the same: stim-pools-not-arch-portable), so a run is reproducible on grace1.

Workers are separate processes (multiprocessing 'spawn', so none inherits TensorFlow's
state) that import only numpy and stim. Chunks come back in submission order, so the batch
order is deterministic. At most `in_flight` chunks are requested ahead of the one being
consumed: the workers sample several times faster than the GPU trains, and without that
limit the sampled chunks pile up in host memory (first full run, 2026-10-09: about 40 GB
within minutes, on the way to a whole 100M-shot epoch, 80 GB).
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
    """Flat detection events [n, n_det] -> [n, n_timesteps, n_positions] uint8, zero where a
    position is not measured at that timestep. Same layout as StudentModels.to_sequence,
    which returns float32; uint8 here because the values are 0/1 (tests check equality)."""
    n_t, n_pos = index_map.shape
    out = np.zeros((det.shape[0], n_t, n_pos), dtype=np.uint8)
    for t in range(n_t):
        cols = index_map[t]
        present = cols >= 0
        out[:, t, present] = det[:, cols[present]]
    return out


def _sample(task):
    """One chunk: (seed, n) -> (x uint8 [n, T, P], flips float32 [n])."""
    seed, n = task
    sampler = _W['circ'].compile_detector_sampler(seed=int(seed))
    det, obs = sampler.sample(int(n), separate_observables=True)
    return _scatter(det.astype(np.uint8), _W['index_map']), obs.reshape(n, -1)[:, 0].astype(np.float32)


def chunk_seed(epoch, chunk, seed_base=SEED_BASE):
    return seed_base + epoch * SEED_EPOCH_STRIDE + chunk


def seed_range(epochs, chunks_per_epoch, seed_base=SEED_BASE):
    """[lo, hi) of every Stim seed a run of `epochs` epochs uses."""
    return seed_base, seed_base + (epochs - 1) * SEED_EPOCH_STRIDE + chunks_per_epoch


def check_seeds(epochs, chunks_per_epoch, seed_base=SEED_BASE):
    """Stop if any seed in configs/partitions.yaml (pools, held-out blocks, seeds used
    elsewhere) lies in the range this run would sample from."""
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
    """Batches of new shots, epoch by epoch.

    batches(epoch) yields (x uint8 [batch, T, P], y float32 [batch, 2]) for
    shots_per_epoch shots; y[:, 0] is the logical flip and y[:, 1] the teacher-logit column
    of the training loss, zero (hard labels only, alpha = 1).
    """

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
        """tf.data over the batches. Keras opens one iterator per epoch, and each new
        iterator samples the next epoch; first_epoch is the epoch the first iterator samples
        (the --initial-epoch of a resumed run)."""
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
