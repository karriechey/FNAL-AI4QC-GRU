<!-- Created: 2026-10-28 | Last updated: 2026-10-28 -->
# kd5_u140 quantization study: plan and pre-registered selection rule

Written 2026-10-28 ~13:00Z, after Phase 1 (PTQ) and before any QAT run started.

## Selection rule (eval block only)

A configuration is acceptable if its eval-block LER <= 1.05 x the FP32 parent's LER and its LER
< MWPM's LER, both on the same 1,800,000 eval-block shots [15.2M, 17M). The candidate is the
acceptable configuration with the fewest total bits (weight bits + state bits), ties broken by
fewer state bits. If nothing below 16 + 16 is acceptable, that is the result.

The held-out blocks (sealed [17M, 19M), seeds 7001 / 7002) are not read anywhere in this study.

## Precision groups and hls4ml knobs (hls4ml main 1259e2b25)

| group | hls4ml precision keys | sweep axis |
|---|---|---|
| GRU input kernel | gru.weight | weights |
| GRU recurrent kernel | gru.recurrent_weight | weights |
| GRU bias (recurrent bias is all zero for reset_after=False) | gru.bias, gru.recurrent_bias | weights |
| Dense head kernel, bias | logit.weight, logit.bias | weights |
| h and r*h | gru.result | state |
| sigmoid and tanh table entries | gru.table (table_size 1024) | state |
| GRU accumulator: products, pre-activations, gate and candidate outputs | gru.accum | held at 24 bits |
| Dense head accumulator and output | logit.accum, logit.result, logit_linear.result | held at 24 bits |
| input | syndrome_sequence.result = ap_uint<1> | exact |

All types AP_RND, AP_SAT. The accumulators are held wide because hls4ml stores the gate outputs
in accum_t: an 8-bit accum_t would also be an 8-bit gate, and the gate would not be swept by the
state axis.

## Integer bits

Phase 1 (PTQ): I = floor(log2 max|v|) + 2 (sign included) + 1 headroom, max over all 1.8M
eval-block shots (results/p1_profile.json).

Phase 2 (QAT): weight groups take the integer bits that minimize the squared error of
ap_fixed<W, I> on the FP32 weights (I in [-3, profiled]); h and r*h take I = 2 and table entries
I = 1 (the smallest that hold [-1, 1] and (-1, 1)); accumulators as Phase 1. Each QAT run's
epoch-0 score is PTQ at its own quantizers, so PTQ and QAT are also compared at equal quantizers.

## QAT

From the FP32 kd5_u140 weights; teacher f5w_p004; BCE to sigmoid(teacher logit), alpha 0, T 1;
fresh Stim shots of the parent's registered seed base 3e12, epochs 80+ (never seen by the parent);
20M shots per epoch, batch 10,000, Adam global clipnorm 0.64, peak LR 3e-4 (1/10 of 3e-3), cosine
to 3e-6 over 25 epochs; eval block after every epoch, best epoch kept; stop after 5 evals without
improvement; NaN or epoch loss > 3x epoch 1 is a failed point, retried once at half LR.

Points: weights {8, 6, 5, 4, 3} at state 16, state {8, 6} at weights 16 (state 10 is above the
PTQ state knee of 8 and is dropped); then at most 6 points around the best combination; 3 seeds
for the candidate and its neighbor one bit lower (seed s = fresh-shot epochs [80 + 25 s, 105 + 25 s)).
