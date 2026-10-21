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
typedef ap_fixed<32,16> input_t;
typedef ap_fixed<32,16> gru_accum_t;
typedef ap_fixed<32,16> layer2_t;
typedef ap_fixed<32,16> gru_weight_t;
typedef ap_fixed<32,16> gru_recurrent_weight_t;
typedef ap_fixed<32,16> gru_bias_t;
typedef ap_fixed<32,16> gru_recurrent_bias_t;
typedef ap_fixed<18,8> gru_table_t;
typedef ap_uint<1> layer2_index;
typedef ap_fixed<72,40> logit_accum_t;
typedef ap_fixed<72,40> result_t;
typedef ap_fixed<32,16> logit_weight_t;
typedef ap_fixed<32,16> logit_bias_t;
typedef ap_uint<1> layer3_index;

// hls-fpga-machine-learning insert emulator-defines


#endif
