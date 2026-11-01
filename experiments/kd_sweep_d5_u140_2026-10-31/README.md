<!-- Created: 2026-10-31 | Last updated: 2026-10-31 -->
# d5 u140 distillation sweep: alpha x T, one seed, FP32

Loss: alpha * BCE(y, z_s) + (1 - alpha) * T^2 * BCE(sigmoid(z_t / T), z_s / T), z = logits
(`src/train_student.py make_distillation_loss`). alpha = 1 is hard labels only, so T has no effect there.

Grid alpha {0, 0.25, 0.5, 0.75, 1} x T {1, 2, 4}: 11 new runs (`configs.tsv`). Reused from the pilot
(`../kd_pilot_p004_2026-10`, same recipe and shots): kd5_u140 (alpha 0, T 1, the baseline) and hl5_u140
(alpha 1).

Fixed, as kd5_u140: teacher f5w_p004 (1 x GRU 280, epoch 80), student 1 x GRU 140 (69,441 params),
initialisation seed 0, 100M fresh Stim shots per epoch from seed base 3e12 (same shots in every run), batch
10,000, Adam global clipnorm 0.64, peak LR 3e-3 cosine to 3e-5 over 80 epochs, TF32 on for training, off for
scoring. Code: the pilot deploy on grace1 (`CODE_COMMIT.txt`).

Checkpoint: lowest pL on the selection block [0, 10M) over epochs 40-80. Scored once on the eval block
[15.2M, 17M), the same 1.8M shots for every run. The sealed block is not read.

Ranking: eval-block pL. Reported per run: selection- and eval-block pL, Wilson 95% interval, ratio to
uncorrelated MWPM and to the teacher, McNemar against kd5_u140 (b = this run right, kd5_u140 wrong).

## Checks before launch

- `test_kd_loss.py`: loss value and gradient match a float64 numpy reference for all 15 (alpha, T) cells
  (max diff 7e-15); alpha 1 is identical for every T and teacher; alpha 0, T 1 equals BCE on the teacher
  probability; soft gradient is 0 at z_s = z_t (`results/test_kd_loss.json`).
- `report.py` on the two pilot cells reproduces their logged numbers (0.680x and 0.684x MWPM, p 0.41).
- Smoke run (alpha 0.5, T 2, 2 epochs of 10M shots): teacher loaded, alpha and T reach the training step,
  selection, eval and copy to keep/ complete.

## Running (grace1)

    cd ~/FNAL-AI4QC-GRU-kd/experiments/kd_sweep_d5_u140_2026-10-31
    setsid nohup bash queue.sh > logs/queue.out 2>&1 < /dev/null &
    ~/tfenv/bin/python report.py

Run folders in /tmp/kchey_kdsweep (/home is full; /tmp clears files after 10 days); the files needed later,
selected weights included, go to keep/. Live log: /tmp/kchey_kdsweep/runs/sweep.log.
