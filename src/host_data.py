# Created: 2026-09-26
# Last modified: 2026-10-01 (HOSTDATA_MAP_PARALLEL, HOSTDATA_PREFETCH, HOSTDATA_GPU_PREFETCH: optional fixed pipeline depths)
"""tf.data input pipelines over host-memory uint8 arrays.

model.fit on a numpy array copies the whole array to the GPU, which fails for the 20M-shot sets.
Here only a few batches are on the GPU at a time. Training batches use a seeded shuffle,
reshuffled each epoch; validation and evaluation keep shot order.
"""
import os

import numpy as np


def to_host_uint8(x):
    # C-contiguous uint8; refuses anything not 0/1
    x = np.asarray(x)
    if x.size and (x.min() < 0 or x.max() > 1):
        raise ValueError(f"expected 0/1 features, got range [{x.min()}, {x.max()}]")
    return np.ascontiguousarray(x, dtype=np.uint8)


def index_dataset(tf, n, batch_size, shuffle_seed=None):
    # batches of row indices; reshuffled each epoch when shuffle_seed is set
    with tf.device('/CPU:0'):
        ds = tf.data.Dataset.range(n)
        if shuffle_seed is not None:
            ds = ds.shuffle(n, seed=shuffle_seed, reshuffle_each_iteration=True)
        return ds.batch(batch_size)


def make_dataset(tf, x_u8, y=None, batch_size=10_000, shuffle_seed=None, to_gpu=True):
    # x_u8 uint8 [N, ...], optional y float32 [N, ...]; yields float32 batches
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
        # Defaults: AUTOTUNE gathers, AUTOTUNE prefetch, 2 batches on the GPU.
        # HOSTDATA_MAP_PARALLEL / HOSTDATA_PREFETCH / HOSTDATA_GPU_PREFETCH fix them to bound
        # host memory. deterministic=True keeps batch order the same for any depth.
        par = int(os.environ.get('HOSTDATA_MAP_PARALLEL', 0)) or tf.data.AUTOTUNE
        pre = int(os.environ.get('HOSTDATA_PREFETCH', 0)) or tf.data.AUTOTUNE
        ds = ds.map(fn, num_parallel_calls=par, deterministic=True)
        ds = ds.prefetch(pre)
    if to_gpu and tf.config.list_logical_devices('GPU'):
        gpre = int(os.environ.get('HOSTDATA_GPU_PREFETCH', 2))
        ds = ds.apply(tf.data.experimental.prefetch_to_device('/GPU:0', buffer_size=gpre))
    return ds
