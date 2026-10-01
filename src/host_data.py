# Created: 2026-09-26
# Last modified: 2026-10-01 (HOSTDATA_MAP_PARALLEL, HOSTDATA_PREFETCH, HOSTDATA_GPU_PREFETCH: optional fixed pipeline depths)
"""Input pipelines that keep the data in host (CPU) memory.

Why this exists
---------------
model.fit / model.predict on a numpy array make Keras copy the whole array to the GPU in
one piece. The 20M-shot training sets are ~12 GB (d=5) and ~30 GB (d=7) as float32, so
with the GPU shared (sglang holds ~91.5 GB of the 97.9 GB card) the copy fails:
"Failed copying input tensor from CPU:0 to GPU:0 ... Dst tensor is not initialized".

Here the arrays stay in host memory as uint8 (the detector events are 0/1, so this is
exact and 4x smaller than float32). A tf.data pipeline, built on the CPU, takes batches
of shot indices, gathers those rows in numpy, casts them to float32, and prefetches the
batches to the GPU, so only a few batches are on the GPU at a time.

Training order: the index dataset is shuffled with a fixed seed and reshuffled every
epoch (reshuffle_each_iteration=True), so each epoch visits every training index exactly
once, in a different order, and the whole sequence is reproducible from the seed.
This order differs from the old pipeline's (Keras' own shuffle of a numpy array), so
new and old runs with the same seed are not bit-for-bit comparable; treat the difference
like seed variance.

Validation and evaluation use the same pipeline without shuffling, in shot order.
"""
import os

import numpy as np


def to_host_uint8(x):
    """Return x as a C-contiguous uint8 array, refusing anything that is not 0/1."""
    x = np.asarray(x)
    if x.size and (x.min() < 0 or x.max() > 1):
        raise ValueError(f"expected 0/1 features, got range [{x.min()}, {x.max()}]")
    return np.ascontiguousarray(x, dtype=np.uint8)


def index_dataset(tf, n, batch_size, shuffle_seed=None):
    """Batches of row indices into an n-row array; shuffled per epoch when a seed is given.

    Built on the CPU. Iterating the returned dataset again starts a new epoch, which is
    reshuffled when shuffle_seed is set. Exposed separately so tests can check exactly the
    index sequence training uses.
    """
    with tf.device('/CPU:0'):
        ds = tf.data.Dataset.range(n)
        if shuffle_seed is not None:
            ds = ds.shuffle(n, seed=shuffle_seed, reshuffle_each_iteration=True)
        return ds.batch(batch_size)


def make_dataset(tf, x_u8, y=None, batch_size=10_000, shuffle_seed=None, to_gpu=True):
    """tf.data pipeline over host-memory arrays.

    x_u8: uint8 array [N, ...] (from to_host_uint8). y: optional float32 array [N, ...].
    Yields float32 x batches (and y batches when y is given). With to_gpu=True the last
    stage prefetches two batches onto /GPU:0 when a GPU is visible.
    """
    n = int(x_u8.shape[0])
    x_shape = tuple(x_u8.shape[1:])
    y_shape = None if y is None else tuple(y.shape[1:])

    def gather_x(idx):
        return x_u8[idx]

    def gather_xy(idx):
        return x_u8[idx], y[idx]

    with tf.device('/CPU:0'):
        ds = index_dataset(tf, n, batch_size, shuffle_seed)
        if y is None:
            def fn(idx):
                xb = tf.numpy_function(gather_x, [idx], tf.uint8)
                xb.set_shape((None,) + x_shape)
                return tf.cast(xb, tf.float32)
        else:
            y_dtype = tf.as_dtype(y.dtype)

            def fn(idx):
                xb, yb = tf.numpy_function(gather_xy, [idx], (tf.uint8, y_dtype))
                xb.set_shape((None,) + x_shape)
                yb.set_shape((None,) + y_shape)
                return tf.cast(xb, tf.float32), yb
        # deterministic=True keeps batch order fixed even with parallel gathers.
        # Pipeline depths. Defaults (env unset) are the ones every earlier run used:
        # AUTOTUNE parallel gathers, AUTOTUNE prefetch, 2 batches prefetched to the GPU.
        # HOSTDATA_MAP_PARALLEL / HOSTDATA_PREFETCH / HOSTDATA_GPU_PREFETCH (integers) fix
        # them, to bound host memory (2026-10-01 memory test). deterministic=True keeps the
        # batch order the same for any depth, so the numbers do not depend on these.
        par = int(os.environ.get('HOSTDATA_MAP_PARALLEL', 0)) or tf.data.AUTOTUNE
        pre = int(os.environ.get('HOSTDATA_PREFETCH', 0)) or tf.data.AUTOTUNE
        ds = ds.map(fn, num_parallel_calls=par, deterministic=True)
        ds = ds.prefetch(pre)
    if to_gpu and tf.config.list_logical_devices('GPU'):
        gpre = int(os.environ.get('HOSTDATA_GPU_PREFETCH', 2))
        ds = ds.apply(tf.data.experimental.prefetch_to_device('/GPU:0', buffer_size=gpre))
    return ds
