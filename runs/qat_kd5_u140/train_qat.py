# Created: 2026-10-28 | Last updated: 2026-10-28
# QAT fine-tune of kd5_u140 on the hls4ml emulator (fq_gru), from the FP32 weights.
# Same teacher (f5w_p004), loss (BCE to the teacher's sigmoid, alpha 0, T 1), fresh-shot stream
# (seed base 3e12, epochs from 80 on: shots the parent never saw), Adam global_clipnorm 0.64,
# cosine from --lr to --lr / 100. Scored on the eval block after every epoch; best epoch kept;
# stop after --patience evals without improvement, or on NaN / divergence.
#   bash pod.sh qat_<tag> python3 train_qat.py --tag w6_s16 --spec-json specs/w6_s16.json
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcommon as Q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--spec-json', required=True)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--shots-per-epoch', type=int, default=20_000_000)
    ap.add_argument('--first-epoch', type=int, default=80)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--gpu-mem-mib', type=int, default=4000)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import tensorflow as tf
    tf.config.experimental.enable_tensor_float_32_execution(False)
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.set_logical_device_configuration(g, [tf.config.LogicalDeviceConfiguration(memory_limit=args.gpu_mem_mib)])
    import fq_gru as F
    import fresh_shots
    from helpers import set_seeds, cosine_lr_schedule
    from train_student import build_teacher_from_weights

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, 'history.jsonl')
    spec = {k: tuple(v) for k, v in json.load(open(args.spec_json)).items()}
    set_seeds(args.seed)
    # init is the parent and the step is deterministic, so a seed is a different block of fresh-shot
    # epochs: seed s trains on epochs [80 + 25 s, 80 + 25 (s + 1)) of the registered base (max 160)
    args.first_epoch += args.epochs * args.seed

    BATCH = 10_000
    chunks = args.shots_per_epoch // 100_000
    fresh_shots.check_seeds(args.first_epoch + args.epochs, chunks, Q.FRESH_SEED_BASE)
    teacher = build_teacher_from_weights(tf, Q.TEACHER_WEIGHTS, [Q.TEACHER_UNITS], Q.D, Q.ROUNDS, Q.P,
                                         expected_params=Q.TEACHER_PARAMS)
    student = F.build_fq(spec)
    F.copy_weights(Q.build_fp32(), student)
    steps = args.epochs * args.shots_per_epoch // BATCH
    opt = tf.keras.optimizers.Adam(learning_rate=cosine_lr_schedule(tf, args.lr, args.lr / 100, steps), global_clipnorm=0.64)


    @tf.function
    def train_step(xb, yb):
        z_t = teacher(xb, training=False)
        with tf.GradientTape() as tape:
            z_s = student(xb, training=True)
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.sigmoid(z_t), z_s, from_logits=True))
        grads = tape.gradient(loss, student.trainable_variables)
        opt.apply_gradients(zip(grads, student.trainable_variables))
        return loss


    d = Q.load_eval()
    x_eval = d['x'].astype(np.float32)


    def evaluate():
        y = F.predict_logits(student, x_eval)
        return y, Q.score(y > 0, d)


    def log(rec):
        with open(log_path, 'a') as f:
            f.write(json.dumps(rec, default=str) + '\n')
        print(json.dumps(rec, default=str), flush=True)


    y0, s0 = evaluate()
    best = {'epoch': 0, 'ler': s0['ler'], 'score': s0}
    np.save(os.path.join(Q.SCRATCH, 'logits', f'qat_{args.tag}_best.npy'), y0.astype(np.float32))
    student.save_weights(os.path.join(args.out, 'best.weights.h5'))
    log({'epoch': 0, 'ler': s0['ler'], 'ler_over_fp32': s0['ler_over_fp32'], 'note': 'start (PTQ at this spec)'})

    fs = fresh_shots.FreshShots(Q.D, Q.P, Q.ROUNDS, args.shots_per_epoch, BATCH, workers=args.workers,
                                seed_base=Q.FRESH_SEED_BASE)
    status, first_loss, since_best = 'finished max epochs', None, 0
    t_start = time.time()
    for e in range(1, args.epochs + 1):
        t0 = time.time()
        losses = []
        for xb, yb in fs.batches(args.first_epoch + e - 1):
            losses.append(float(train_step(tf.convert_to_tensor(xb, tf.float32), tf.convert_to_tensor(yb))))
        mean_loss = float(np.mean(losses))
        first_loss = first_loss or mean_loss
        if not np.isfinite(mean_loss) or mean_loss > 3 * first_loss:
            status = 'failed: loss NaN' if not np.isfinite(mean_loss) else 'failed: loss diverged (> 3x epoch-1 loss)'
            log({'epoch': e, 'loss': mean_loss, 'status': status})
            break
        y, s = evaluate()
        improved = s['ler'] < best['ler']
        if improved:
            best = {'epoch': e, 'ler': s['ler'], 'score': s}
            student.save_weights(os.path.join(args.out, 'best.weights.h5'))
            np.save(os.path.join(Q.SCRATCH, 'logits', f'qat_{args.tag}_best.npy'), y.astype(np.float32))
            since_best = 0
        else:
            since_best += 1
        log({'epoch': e, 'loss': mean_loss, 'ler': s['ler'], 'ler_over_fp32': s['ler_over_fp32'],
             'ler_over_mwpm': s['ler_over_mwpm'], 'improved': improved, 'epoch_s': round(time.time() - t0, 1),
             'lr': float(opt.learning_rate(opt.iterations)) if callable(opt.learning_rate) else None})
        if since_best >= args.patience:
            status = f'early stop: no improvement for {args.patience} evals'
            break
    fs.close()
    Q.dump({'tag': args.tag, 'spec': spec, 'args': vars(args), 'status': status, 'best_epoch': best['epoch'],
            'score': best['score'], 'start_score': s0, 'train_s': round(time.time() - t_start, 1)},
           os.path.join(args.out, 'result.json'))
    print('DONE', status, best['epoch'], best['ler'], flush=True)


if __name__ == '__main__':
    # FreshShots workers are spawned and import this file
    main()
