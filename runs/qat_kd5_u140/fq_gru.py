# Created: 2026-10-28 | Last updated: 2026-10-28
# Fake-quantized GRU(reset_after=False) + Dense head that follows the hls4ml Vivado kernel
# (nnet_recurrent.h gru_static / gru_resource, nnet_recr_activations.h) at main 1259e2b25:
#   accum_t   input and recurrent products, pre-activations, and the gate / candidate outputs
#   result_t  h and r*h
#   table_t   sigmoid table on [-8, 8), tanh table on [-4, 4), index trunc(x * N / range) + N / 2
# Every ap_fixed cast is AP_RND, AP_SAT. Quantizers pass gradients straight through (QAT).
# spec: {group: (W, I)} as in qcommon.spec_for; a missing group stays float. tables=False uses
# exact sigmoid / tanh.
import numpy as np
import tensorflow as tf

N_T, N_POS = 6, 24


def quantize(x, w, i):
    f = w - i
    s = 2.0 ** f
    lo, hi = -2.0 ** (i - 1), 2.0 ** (i - 1) - 1.0 / s
    y = tf.clip_by_value(tf.floor(x * s + 0.5) / s, lo, hi)
    return x + tf.stop_gradient(y - x)


def quantize_np(x, w, i):
    f = w - i
    s = 2.0 ** f
    return np.clip(np.floor(np.asarray(x, np.float64) * s + 0.5) / s, -2.0 ** (i - 1), 2.0 ** (i - 1) - 1.0 / s)


def table_values(fn, half_range, n, table_spec):
    # init_sigmoid_table / init_tanh_table: float grid, float function, cast to table_t
    ii = np.arange(n, dtype=np.float32)
    x = np.float32(2 * half_range) * (ii - np.float32(n / 2.0)) / np.float32(n)
    v = fn(x).astype(np.float32)
    return quantize_np(v, *table_spec).astype(np.float32) if table_spec else v


class FQGRU(tf.keras.layers.Layer):
    def __init__(self, units, spec, tables=True, table_size=1024, **kw):
        super().__init__(**kw)
        self.units, self.spec, self.tables, self.table_size = units, dict(spec or {}), tables, table_size
        tspec = self.spec.get('gru_table')
        sig = lambda v: np.float32(1.0) / (np.float32(1.0) + np.exp(-v))
        self.sig_table = tf.constant(table_values(sig, 8.0, table_size, tspec))
        self.tanh_table = tf.constant(table_values(np.tanh, 4.0, table_size, tspec))

    def build(self, input_shape):
        n_in, n = int(input_shape[-1]), self.units
        self.kernel = self.add_weight(name='kernel', shape=(n_in, 3 * n), initializer='zeros')
        self.recurrent_kernel = self.add_weight(name='recurrent_kernel', shape=(n, 3 * n), initializer='zeros')
        self.bias = self.add_weight(name='bias', shape=(3 * n,), initializer='zeros')

    def q(self, group, x):
        s = self.spec.get(group)
        return quantize(x, *s) if s else x

    def lookup(self, x, table, half_range, exact):
        if not self.tables:
            return exact(x)
        n = self.table_size
        scaled = x * (n / (2.0 * half_range))
        idx = tf.cast(tf.clip_by_value(tf.sign(scaled) * tf.floor(tf.abs(scaled)) + n // 2, 0, n - 1), tf.int32)
        smooth = exact(x)
        return smooth + tf.stop_gradient(tf.gather(table, idx) - smooth)

    def call(self, x, return_internals=False):
        n = self.units
        W = self.q('gru_weight', self.kernel)
        U = self.q('gru_recurrent_weight', self.recurrent_kernel)
        b = self.q('gru_bias', self.bias)
        acc = lambda v: self.q('gru_accum', v)
        st = lambda v: self.q('gru_state', v)
        h = tf.zeros((tf.shape(x)[0], n), x.dtype)
        taps = []
        for t in range(x.shape[1]):
            xin = acc(tf.matmul(x[:, t], W) + b)
            rec_zr = acc(tf.matmul(h, U[:, :2 * n]))
            pre_zr = acc(xin[:, :2 * n] + rec_zr)
            zr = acc(self.lookup(pre_zr, self.sig_table, 8.0, tf.sigmoid))
            z, r = zr[:, :n], zr[:, n:]
            h_reset = st(r * h)
            rec_h = acc(tf.matmul(h_reset, U[:, 2 * n:]))
            pre_h = acc(xin[:, 2 * n:] + rec_h)
            hh = acc(self.lookup(pre_h, self.tanh_table, 4.0, tf.tanh))
            h = st(hh * (1.0 - z) + h * z)
            if return_internals:
                taps.append({'xin': xin, 'rec_zr': rec_zr, 'pre_zr': pre_zr, 'zr': zr, 'h_reset': h_reset,
                             'rec_h': rec_h, 'pre_h': pre_h, 'hh': hh, 'h': h})
        return (h, taps) if return_internals else h


class FQDense(tf.keras.layers.Layer):
    def __init__(self, spec, **kw):
        super().__init__(**kw)
        self.spec = dict(spec or {})

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(int(input_shape[-1]), 1), initializer='zeros')
        self.bias = self.add_weight(name='bias', shape=(1,), initializer='zeros')

    def call(self, h):
        q = lambda g, v: quantize(v, *self.spec[g]) if self.spec.get(g) else v
        return q('logit_accum', tf.matmul(h, q('logit_weight', self.kernel)) + q('logit_bias', self.bias))


def build_fq(spec, tables=True, units=140, table_size=1024):
    inp = tf.keras.Input(shape=(N_T, N_POS), name='syndrome_sequence')
    h = FQGRU(units, spec, tables=tables, table_size=table_size, name='gru')(inp)
    return tf.keras.Model(inp, FQDense(spec, name='logit')(h), name='fq_gru')


def copy_weights(src, dst):
    # src: Keras GRU + Dense ('gru', 'logit'), or another fq model
    dst.get_layer('gru').set_weights(src.get_layer('gru').get_weights()[:3])
    dst.get_layer('logit').set_weights(src.get_layer('logit').get_weights())


def predict_logits(model, x, batch=20000):
    @tf.function(reduce_retracing=True)
    def f(xb):
        return model(xb, training=False)
    out = [f(tf.convert_to_tensor(x[a:a + batch], tf.float32)).numpy().reshape(-1) for a in range(0, len(x), batch)]
    return np.concatenate(out)
