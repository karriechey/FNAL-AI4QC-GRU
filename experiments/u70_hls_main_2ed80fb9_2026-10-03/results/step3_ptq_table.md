# u70 GRU PTQ width sweep (C simulation only, not synthesized)

hls4ml `2ed80fb9a6d5150a7f1ec21a8cf6564900244a0e`; 1,800,000 shots of the FORMAL evaluation block [15.2M, 17.0M) (EAF pool); reference = FP32 Keras on the same shots. Integer bits per type from profiling (results/step3_profiled_ranges.json); b = Keras right & HLS wrong, c = HLS right & Keras wrong.

| mode | W | p_L HLS | p_L Keras | b | c | n_disc | McNemar p | ΔLER 95% CI | same as FP32? | capped types | h trace mean err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| trn_wrap | 24 | 0.007024 | 0.007011 | 137 | 113 | 250 | 0.146 | [-3.88e-06, +3.06e-05] | yes | none | 2.83e-03 |
| trn_wrap | 18 | 0.007096 | 0.007011 | 585 | 431 | 1016 | 1.52e-06 | [+5.08e-05, +1.20e-04] | no | none | 1.26e-02 |
| trn_wrap | 16 | 0.007521 | 0.007011 | 2258 | 1340 | 3598 | 2.52e-53 | [+4.45e-04, +5.75e-04] | no | none | 3.97e-02 |
| trn_wrap | 12 | 0.137413 | 0.007011 | 239729 | 5005 | 244734 | 0 | [+1.30e-01, +1.31e-01] | no | none | 2.68e-01 |
| trn_wrap | 10 | 0.207941 | 0.007011 | 367060 | 5386 | 372446 | 0 | [+2.00e-01, +2.02e-01] | no | none | 4.77e-01 |
| trn_wrap | 8 | 0.253417 | 0.007011 | 449256 | 5725 | 454981 | 0 | [+2.46e-01, +2.47e-01] | no | gru_accum | 7.18e-01 |
| rnd_sat | 24 | 0.007030 | 0.007011 | 135 | 100 | 235 | 0.0263 | [+2.75e-06, +3.61e-05] | no | none | 2.78e-03 |
| rnd_sat | 18 | 0.007010 | 0.007011 | 119 | 120 | 239 | 1 | [-1.74e-05, +1.63e-05] | yes | none | 2.80e-03 |
| rnd_sat | 16 | 0.007023 | 0.007011 | 205 | 183 | 388 | 0.286 | [-9.23e-06, +3.37e-05] | yes | none | 4.12e-03 |
| rnd_sat | 12 | 0.007831 | 0.007011 | 3457 | 1981 | 5438 | 4.23e-90 | [+7.40e-04, +9.00e-04] | no | none | 4.92e-02 |
| rnd_sat | 10 | 0.085295 | 0.007011 | 146595 | 5683 | 152278 | 0 | [+7.79e-02, +7.87e-02] | no | none | 3.22e-01 |
| rnd_sat | 8 | 0.260658 | 0.007011 | 463055 | 6489 | 469544 | 0 | [+2.53e-01, +2.54e-01] | no | gru_accum | 4.69e-01 |

trn_wrap: smallest width indistinguishable from FP32 (it and every wider swept width p >= 0.05): 24

rnd_sat: smallest width indistinguishable from FP32 (it and every wider swept width p >= 0.05): None

Limiting type at trn_wrap_w18 (200,000 shots, one group restored to ap_fixed<32,16> at a time; decision flips vs Keras):
- none restored: 97 flips, h trace mean err 1.26e-02
- gru weights+biases: 102 flips, h trace mean err 1.27e-02
- gru accum: 36 flips, h trace mean err 2.87e-03
- gru result (h): 97 flips, h trace mean err 1.26e-02
- gru activation table: 97 flips, h trace mean err 1.26e-02
- logit (all types): 95 flips, h trace mean err 1.26e-02
- input: 97 flips, h trace mean err 1.26e-02

Limiting type at rnd_sat_w24 (200,000 shots, one group restored to ap_fixed<32,16> at a time; decision flips vs Keras):
- none restored: 27 flips, h trace mean err 2.78e-03
- gru weights+biases: 32 flips, h trace mean err 2.79e-03
- gru accum: 31 flips, h trace mean err 2.83e-03
- gru result (h): 27 flips, h trace mean err 2.79e-03
- gru activation table: 27 flips, h trace mean err 2.78e-03
- logit (all types): 27 flips, h trace mean err 2.78e-03
- input: 27 flips, h trace mean err 2.78e-03

Integer bits used: input 2, gru_weight 6, gru_recurrent_weight 7, gru_bias 5, gru_recurrent_bias 5, gru_accum 8, gru_result 2, gru_table 1, logit_weight 4, logit_bias 1, logit_accum 6, logit_result 6
