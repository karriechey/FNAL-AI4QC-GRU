# Created: 2026-10-31 | Last updated: 2026-10-31
# Checks make_distillation_loss (src/train_student.py) against a float64 numpy reference before the
# sweep: value and gradient w.r.t. the student logit for every (alpha, T) of the grid, alpha=1 independent
# of T and of the teacher, alpha=0 T=1 equal to plain BCE on the teacher probability, the soft gradient
# zero at z_student = z_teacher, and the T^2 factor (soft gradient = T (sigmoid(z_s/T) - sigmoid(z_t/T))).
# CPU only.
#   ~/tfenv/bin/python test_kd_loss.py <src dir>
import json
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, sys.argv[1])
import numpy as np
import tensorflow as tf
from train_student import make_distillation_loss

ALPHAS, TEMPS = (0.0, 0.25, 0.5, 0.75, 1.0), (1.0, 2.0, 4.0)
rng = np.random.default_rng(0)
n = 4096
z_s = rng.normal(0, 4, n)
z_t = rng.normal(0, 4, n)
y = (rng.random(n) < 0.3).astype(np.float64)
sig = lambda v: 1 / (1 + np.exp(-v))


def bce_logits(target, z):
    return np.maximum(z, 0) - z * target + np.log1p(np.exp(-np.abs(z)))


def reference(a, T):
    hard = bce_logits(y, z_s)
    soft = T * T * bce_logits(sig(z_t / T), z_s / T)
    grad = a * (sig(z_s) - y) + (1 - a) * T * (sig(z_s / T) - sig(z_t / T))
    return a * hard + (1 - a) * soft, grad


def tf_loss(a, T, zs, zt):
    zs_v = tf.Variable(zs.reshape(-1, 1))
    y_true = tf.constant(np.stack([y, zt], axis=1))
    with tf.GradientTape() as tape:
        per_shot = make_distillation_loss(a, T)(y_true, zs_v)
    return per_shot.numpy().reshape(-1), tape.gradient(per_shot, zs_v).numpy().reshape(-1)


checks, ok = [], True
for a in ALPHAS:
    for T in TEMPS:
        val, grad = tf_loss(a, T, z_s, z_t)
        ref_val, ref_grad = reference(a, T)
        c = {'alpha': a, 'T': T, 'max_abs_value_diff': float(np.abs(val - ref_val).max()),
             'max_abs_grad_diff': float(np.abs(grad - ref_grad).max())}
        c['ok'] = c['max_abs_value_diff'] < 1e-9 and c['max_abs_grad_diff'] < 1e-9
        ok &= c['ok']
        checks.append(c)

hard_only = [tf_loss(1.0, T, z_s, z_t)[0] for T in TEMPS] + [tf_loss(1.0, 1.0, z_s, -z_t)[0]]
alpha1_independent = all(np.array_equal(hard_only[0], h) for h in hard_only[1:])
alpha0_T1 = float(np.abs(tf_loss(0.0, 1.0, z_s, z_t)[0] - bce_logits(sig(z_t), z_s)).max())
zero_grad = max(float(np.abs(tf_loss(0.0, T, z_t, z_t)[1]).max()) for T in TEMPS)
ok &= alpha1_independent and alpha0_T1 < 1e-9 and zero_grad < 1e-12

out = {'checks': checks, 'alpha1_same_for_every_T_and_teacher': alpha1_independent,
       'alpha0_T1_vs_bce_on_teacher_prob_max_diff': alpha0_T1,
       'soft_grad_at_z_student_eq_z_teacher_max': zero_grad, 'passed': bool(ok)}
print(json.dumps(out, indent=1))
sys.exit(0 if ok else 1)
