<!-- Created: 2026-11-01 | Last updated: 2026-11-01 -->
# d5 round-2 distillation: 1 x GRU 70 (20,021 params) from the best sweep student

Starts by itself when the alpha x T sweep (`../kd_sweep_d5_u140_2026-10-31`) finishes (`after_sweep.sh`).

## Rules (fixed before the sweep finished)

- Best sweep cell: rank 1 by eval-block pL (`choose.py`). Round 2 runs only if it beats uncorrelated MWPM
  (eval x MWPM < 1).
- Round-2 teacher: the best cell's u140 student, with its (alpha, T). If the best cell is alpha 1 (hard
  labels), teacher and (alpha, T) come from the best alpha < 1 cell instead.
- Teacher check as in the pilot (`ta_check.sh`): sha256 copy, eval-block errors equal to the sweep table,
  training-step logits equal to the eval dump on 100k shots, TF32 rule.
- One student, 1 x GRU 70 (20,021 params, 3.5x fewer than u140), from the round-2 teacher. Chosen over
  other widths in 6k-35k: the pilot's kd5_u70 (directly from f5w_p004, alpha 0, T 1, same shots) already
  beats MWPM (0.810x) and is the control; u70 also has existing hls4ml synthesis to compare against.
- Recipe, shots (seed base 3e12), checkpoint selection and eval block as the sweep.
- `report_r2.py`: eval pL, x MWPM, McNemar vs kd5_u140 and vs kd5_u70.

## Next: QAT

The QAT procedure of `runs/qat_kd5_u140` (fake-quant emulator bit-exact with hls4ml C-sim, fresh-shot KD,
same selection rule) on:
- the round-2 u70 student, if its FP32 eval pL beats MWPM;
- the best u140 sweep student, compared against kd5_u140's QAT result (w12_s6).

## Files

`after_sweep.sh` (chain), `choose.py`, `ta_check.sh`, `run_r2.sh`, `report_r2.py`; `STATUS` (progress),
`choice.json`, `ta_teacher.sh`, `teacher_check/`, `keep/`, `results/` are written on grace1.
