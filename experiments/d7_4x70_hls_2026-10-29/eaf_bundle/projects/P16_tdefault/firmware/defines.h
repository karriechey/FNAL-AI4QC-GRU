#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,2,AP_RND,AP_SAT,0> input_t;
typedef ap_fixed<16,7,AP_RND,AP_SAT,0> gru_accum_t;
typedef ap_fixed<16,1,AP_RND,AP_SAT,0> layer2_t;
typedef ap_fixed<16,5,AP_RND,AP_SAT,0> gru_weight_t;
typedef ap_fixed<16,4,AP_RND,AP_SAT,0> gru_recurrent_weight_t;
typedef ap_fixed<16,4,AP_RND,AP_SAT,0> gru_bias_t;
typedef ap_fixed<16,4,AP_RND,AP_SAT,0> gru_recurrent_bias_t;
typedef ap_fixed<18,8> gru_table_t;
typedef ap_uint<1> layer2_index;
typedef ap_fixed<16,6,AP_RND,AP_SAT,0> gru_1_accum_t;
typedef ap_fixed<16,1,AP_RND,AP_SAT,0> layer3_t;
typedef ap_fixed<16,4,AP_RND,AP_SAT,0> gru_1_weight_t;
typedef ap_fixed<16,6,AP_RND,AP_SAT,0> gru_1_recurrent_weight_t;
typedef ap_fixed<16,3,AP_RND,AP_SAT,0> gru_1_bias_t;
typedef ap_fixed<16,3,AP_RND,AP_SAT,0> gru_1_recurrent_bias_t;
typedef ap_fixed<18,8> gru_1_table_t;
typedef ap_uint<1> layer3_index;
typedef ap_fixed<16,7,AP_RND,AP_SAT,0> gru_2_accum_t;
typedef ap_fixed<16,2,AP_RND,AP_SAT,0> layer4_t;
typedef ap_fixed<16,4,AP_RND,AP_SAT,0> gru_2_weight_t;
typedef ap_fixed<16,4,AP_RND,AP_SAT,0> gru_2_recurrent_weight_t;
typedef ap_fixed<16,2,AP_RND,AP_SAT,0> gru_2_bias_t;
typedef ap_fixed<16,2,AP_RND,AP_SAT,0> gru_2_recurrent_bias_t;
typedef ap_fixed<18,8> gru_2_table_t;
typedef ap_uint<1> layer4_index;
typedef ap_fixed<16,7,AP_RND,AP_SAT,0> gru_3_accum_t;
typedef ap_fixed<16,2,AP_RND,AP_SAT,0> layer5_t;
typedef ap_fixed<16,4,AP_RND,AP_SAT,0> gru_3_weight_t;
typedef ap_fixed<16,5,AP_RND,AP_SAT,0> gru_3_recurrent_weight_t;
typedef ap_fixed<16,2,AP_RND,AP_SAT,0> gru_3_bias_t;
typedef ap_fixed<16,2,AP_RND,AP_SAT,0> gru_3_recurrent_bias_t;
typedef ap_fixed<18,8> gru_3_table_t;
typedef ap_uint<1> layer5_index;
typedef ap_fixed<16,6,AP_RND,AP_SAT,0> logit_accum_t;
typedef ap_fixed<16,6,AP_RND,AP_SAT,0> result_t;
typedef ap_fixed<16,3,AP_RND,AP_SAT,0> logit_weight_t;
typedef ap_fixed<16,1,AP_RND,AP_SAT,0> logit_bias_t;
typedef ap_uint<1> layer6_index;

// hls-fpga-machine-learning insert emulator-defines


#endif
