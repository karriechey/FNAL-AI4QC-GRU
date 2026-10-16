# Created: 2026-10-03 | Last updated: 2026-10-03
"""Step 4: does hls4ml main 2ed80fb9 parse a QKeras QGRU (reset_after=False), and does its
C simulation match Keras? Small check, no training.

Model: input (6, 24) -> QGRU(8 units, reset_after=False) -> QDense(1), random weights with a
fixed seed, quantized_bits(8, 0, alpha=1) for kernel / recurrent kernel / bias and 8-bit
activations. Two activation variants, because hls4ml rejects some QKeras activations in
recurrent layers:
    hard   quantized_tanh(8) / hard_sigmoid-style quantized_sigmoid(8)  (QKeras defaults)
    real   quantized_tanh(8, use_real_tanh=True) / quantized_sigmoid(8, use_real_sigmoid=True)
    float_act  tanh / sigmoid in float, weights still quantized (diagnostic)
    float_act_wide  float_act with the non-quantizer types at ap_fixed<32,16> (diagnostic)
    float_act_wide_prequant  float_act_wide with weights set to their quantized values first
Each variant: convert (Vivado, Resource, ReuseFactor 1, static, io_parallel, xcvu13p, 5 ns,
precision from the QKeras quantizers via config_from_keras_model), check the generated
reset_after flag and bias sizes, compile, and compare hls_model.predict with Keras on the
first 10,000 shots of the evaluation block.

Usage (grace1): ~/hls4ml_main_env/bin/python step4_qgru_check.py
Writes: results/step4_qgru_check.json
"""
import json
import os
import re
import traceback

import numpy as np

import u70_common as U

# QKeras 0.9 recurrent cells call tf nest.is_sequence, removed in TF 2.15; alias it to the
# renamed is_nested (same function) before any QGRU is built.
from tensorflow.python.util import nest as _nest
if not hasattr(_nest, 'is_sequence') and hasattr(_nest, 'is_nested'):
    _nest.is_sequence = _nest.is_nested
import tensorflow as tf
import qkeras
from qkeras import QDense, QGRU, quantized_bits, quantized_sigmoid, quantized_tanh

env = U.environment_record()
env['qkeras'] = qkeras.__version__ if hasattr(qkeras, '__version__') else 'unknown'
d = U.load_eval(10000)
x = d['x']
UNITS = 8
q = lambda: quantized_bits(8, 0, alpha=1)
VARIANTS = {
    'hard': (quantized_tanh(8), quantized_sigmoid(8)),
    'real': (quantized_tanh(8, use_real_tanh=True), quantized_sigmoid(8, use_real_sigmoid=True)),
    # float activations, quantized weights: separates the weight quantizers from the activations
    'float_act': ('tanh', 'sigmoid'),
    # same as float_act, with every type that no QKeras quantizer sets (input, accum, result,
    # recurrent bias, activation table, Dense accum/result) widened to ap_fixed<32,16> and a
    # 4096-entry table: if C sim then matches Keras closely, the weight quantizers are handled
    # exactly and the remaining mismatch comes from those default types
    'float_act_wide': ('tanh', 'sigmoid'),
    # float_act_wide with the QGRU kernel / recurrent kernel / bias replaced by their QKeras-quantized
    # values before conversion. hls4ml main 2ed80fb9 does not apply the QGRU weight quantizers to the
    # weight data (GRU.initialize in hls4ml/model/layers.py calls add_weights without the quantizer,
    # unlike Dense), so the raw float weights are cast to ap_fixed<8,1> by truncation instead of
    # QKeras rounding. Pre-quantized weights make that cast exact.
    'float_act_wide_prequant': ('tanh', 'sigmoid'),
}
WIDE = 'ap_fixed<32,16>'
out = {'environment': env, 'units': UNITS, 'shots': [U.EVAL_START, U.EVAL_START + len(x)], 'variants': {}}
for name, (act, recr_act) in VARIANTS.items():
    rec = {'activation': str(act), 'recurrent_activation': str(recr_act)}
    try:
        tf.keras.utils.set_random_seed(7)
        inp = tf.keras.Input(shape=(6, 24), name='syndrome_sequence')
        h = QGRU(UNITS, activation=act, recurrent_activation=recr_act, reset_after=False,
                 kernel_quantizer=q(), recurrent_quantizer=q(), bias_quantizer=q(), name='qgru')(inp)
        model = tf.keras.Model(inp, QDense(1, kernel_quantizer=q(), bias_quantizer=q(), name='logit')(h))
        rec['keras_params'] = int(model.count_params())
        rec['keras_bias_shape'] = list(model.get_layer('qgru').get_weights()[2].shape)
        if name.endswith('_prequant'):
            g = model.get_layer('qgru')
            g.set_weights([np.asarray(qz(w)) for qz, w in zip(
                (g.kernel_quantizer_internal, g.recurrent_quantizer_internal, g.bias_quantizer_internal), g.get_weights())])
        y_keras = model.predict(x, batch_size=10000, verbose=0).reshape(-1)
        import hls4ml
        cfg = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Vivado')
        cfg['Model']['Strategy'] = 'Resource'
        cfg['LayerName']['qgru']['static'] = True
        if name.startswith('float_act_wide'):
            cfg['LayerName']['syndrome_sequence']['Precision']['result'] = WIDE
            for key in ('accum', 'result', 'recurrent_bias', 'table'):
                cfg['LayerName']['qgru']['Precision'][key] = WIDE
            cfg['LayerName']['qgru']['TableSize'] = 4096
            for key in ('accum', 'result'):
                cfg['LayerName']['logit']['Precision'][key] = WIDE
        rec['config'] = cfg
        prj = os.path.join(U.HERE, f'hls_step4_qgru_{name}')
        hm = hls4ml.converters.convert_from_keras_model(
            model, hls_config=cfg, output_dir=prj, project_name='myproject', backend='Vivado',
            io_type=U.IO_TYPE, part=U.PART, clock_period=U.CLOCK_NS)
        rec['parsed'] = True
        hm.write()
        params_h = open(os.path.join(prj, 'firmware', 'parameters.h')).read()
        rec['generated_reset_after'] = re.findall(r'static const bool reset_after = (\w+);', params_h)
        gl = [l for l in hm.get_layers() if l.class_name == 'GRU'][0]
        rec['hls_weight_shapes'] = {k: list(np.shape(w.data)) for k, w in gl.weights.items()}
        rec['hls_layer_precisions'] = {l.name: {k: str(v.precision) for k, v in l.get_layer_precision().items()}
                                       for l in hm.get_layers()}
        hm.compile()
        y_hls = np.asarray(hm.predict(np.ascontiguousarray(x)), float).reshape(-1)
        diff = np.abs(y_hls - y_keras)
        rec.update({'csim_ran': True, 'max_abs_diff': float(diff.max()), 'mean_abs_diff': float(diff.mean()),
                    'logit_exact_match_rate': float((y_hls == y_keras).mean()),
                    'decision_match_rate': float(((y_hls > 0) == (y_keras > 0)).mean()),
                    'keras_logit_range': [float(y_keras.min()), float(y_keras.max())]})
    except Exception as e:  # report the failure, keep the other variant
        rec.update({'error': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()[-3000:]})
        rec.setdefault('parsed', False)
    out['variants'][name] = rec
    print(name, json.dumps({k: v for k, v in rec.items() if k not in ('config', 'traceback', 'hls_layer_precisions')},
                           default=str), flush=True)

# ---- control: plain Keras GRU with the same quantized weights, same wide conversion ----------
# If QKeras and Keras agree in Keras and both hls4ml projects show the same error against
# Keras, the QGRU path is as exact as the plain GRU path and the error is the fixed-point
# activation tables, not the QKeras handling.
try:
    tf.keras.utils.set_random_seed(7)
    inp = tf.keras.Input(shape=(6, 24), name='syndrome_sequence')
    qm = tf.keras.Model(inp, QDense(1, kernel_quantizer=q(), bias_quantizer=q(), name='logit')(
        QGRU(UNITS, activation='tanh', recurrent_activation='sigmoid', reset_after=False,
             kernel_quantizer=q(), recurrent_quantizer=q(), bias_quantizer=q(), name='qgru')(inp)))
    qg, qd = qm.get_layer('qgru'), qm.get_layer('logit')
    wq = [np.asarray(qz(w)) for qz, w in zip((qg.kernel_quantizer_internal, qg.recurrent_quantizer_internal,
                                                 qg.bias_quantizer_internal), qg.get_weights())]
    dq = [np.asarray(qz(w)) for qz, w in zip((qd.kernel_quantizer_internal, qd.bias_quantizer_internal), qd.get_weights())]
    inp2 = tf.keras.Input(shape=(6, 24), name='syndrome_sequence')
    gm = tf.keras.Model(inp2, tf.keras.layers.Dense(1, name='logit')(
        tf.keras.layers.GRU(UNITS, reset_after=False, name='qgru')(inp2)))
    gm.get_layer('qgru').set_weights(wq); gm.get_layer('logit').set_weights(dq)
    yq = qm.predict(x, batch_size=10000, verbose=0).reshape(-1)
    yg = gm.predict(x, batch_size=10000, verbose=0).reshape(-1)
    ctl = {'keras_qgru_vs_keras_gru_same_quantized_weights_max_abs_diff': float(np.abs(yq - yg).max())}
    import hls4ml
    cfg = hls4ml.utils.config_from_keras_model(gm, granularity='name', default_precision=WIDE, backend='Vivado')
    cfg['Model']['Strategy'] = 'Resource'
    cfg['LayerName']['qgru']['static'] = True
    cfg['LayerName']['qgru']['TableSize'] = 4096
    cfg['LayerName']['qgru']['Precision']['table'] = WIDE
    hm = hls4ml.converters.convert_from_keras_model(
        gm, hls_config=cfg, output_dir=os.path.join(U.HERE, 'hls_step4_control_keras_gru'), project_name='myproject',
        backend='Vivado', io_type=U.IO_TYPE, part=U.PART, clock_period=U.CLOCK_NS)
    hm.compile()
    yh = np.asarray(hm.predict(np.ascontiguousarray(x)), float).reshape(-1)
    ctl.update({'keras_gru_hls_max_abs_diff': float(np.abs(yh - yg).max()),
                'keras_gru_hls_mean_abs_diff': float(np.abs(yh - yg).mean()),
                'keras_gru_hls_decision_match_rate': float(((yh > 0) == (yg > 0)).mean())})
except Exception as e:
    ctl = {'error': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()[-3000:]}
out['control_plain_keras_gru'] = ctl
print('control', json.dumps({k: v for k, v in ctl.items() if k != 'traceback'}), flush=True)
os.makedirs(os.path.join(U.HERE, 'results'), exist_ok=True)
json.dump(out, open(os.path.join(U.HERE, 'results', 'step4_qgru_check.json'), 'w'), indent=1, default=str)
