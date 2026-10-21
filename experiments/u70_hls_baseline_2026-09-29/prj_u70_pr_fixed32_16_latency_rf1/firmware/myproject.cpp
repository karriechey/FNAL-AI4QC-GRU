#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    input_t syndrome_sequence[6*24],
    result_t layer3_out[1]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=syndrome_sequence complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=syndrome_sequence,layer3_out 
    #pragma HLS PIPELINE

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<gru_weight_t, 5040>(w2, "w2.txt");
        nnet::load_weights_from_txt<gru_recurrent_weight_t, 14700>(wr2, "wr2.txt");
        nnet::load_weights_from_txt<gru_bias_t, 210>(b2, "b2.txt");
        nnet::load_weights_from_txt<gru_recurrent_bias_t, 210>(br2, "br2.txt");
        nnet::load_weights_from_txt<logit_weight_t, 70>(w3, "w3.txt");
        nnet::load_weights_from_txt<logit_bias_t, 1>(b3, "b3.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[70];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    nnet::gru_stack<input_t, layer2_t, config2>(syndrome_sequence, layer2_out, w2, wr2, b2, br2); // gru

    nnet::dense<layer2_t, result_t, config3>(layer2_out, layer3_out, w3, b3); // logit

}

