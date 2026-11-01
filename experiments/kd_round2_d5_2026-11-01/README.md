<!-- Created: 2026-11-01 | Last updated: 2026-11-01 -->
# d5 round-2 distillation: 1 x GRU 34-96 (6k-35k params) from the best sweep student

Starts by itself when the alpha x T sweep (`../kd_sweep_d5_u140_2026-10-31`) finishes (`after_sweep.sh`).

## Rules (fixed before the sweep finished)

- Best sweep cell: rank 1 by eval-block pL (`choose.py`). Round 2 runs only if it beats uncorrelated MWPM
  (eval x MWPM < 1).
- Round-2 teacher: the best cell's u140 student, with its (alpha, T). If the best cell is alpha 1 (hard
  labels), teacher and (alpha, T) come from the best alpha < 1 cell instead.
- Teacher check as in the pilot (`ta_check.sh`): sha256 copy, eval-block errors equal to the sweep table,
  training-step logits equal to the eval dump on 100k shots, TF32 rule.
- Students: 1 x GRU 34 / 48 / 70 / 96 (6,053 / 10,561 / 20,021 / 34,945 params), two arms per width:
  ta (from the round-2 teacher) and f5w (from f5w_p004 directly, the control). The pilot's kd5_u70 is the
  f5w u70 cell when (alpha, T) = (0, 1).
- Recipe, shots (seed base 3e12), checkpoint selection and eval block as the sweep.
- `report_r2.py`: eval pL, x MWPM, McNemar vs kd5_u140, and ta vs f5w per width.

## Next: QAT

The QAT procedure of `runs/qat_kd5_u140` (fake-quant emulator bit-exact with hls4ml C-sim, fresh-shot KD,
same selection rule) on:
- the smallest round-2 student whose FP32 eval pL beats MWPM;
- the best u140 sweep student, compared against kd5_u140's QAT result (w12_s6).

## Files

`after_sweep.sh` (chain), `choose.py`, `ta_check.sh`, `run_r2.sh`, `report_r2.py`; `STATUS` (progress),
`choice.json`, `ta_teacher.sh`, `teacher_check/`, `keep/`, `results/` are written on grace1.
