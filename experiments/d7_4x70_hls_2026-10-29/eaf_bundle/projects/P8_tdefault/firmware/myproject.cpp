#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    input_t syndrome_sequence[8*48],
    result_t layer6_out[1]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=syndrome_sequence complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=syndrome_sequence,layer6_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<gru_weight_t, 10080>(w2, "w2.txt");
        nnet::load_weights_from_txt<gru_recurrent_weight_t, 14700>(wr2, "wr2.txt");
        nnet::load_weights_from_txt<gru_bias_t, 210>(b2, "b2.txt");
        nnet::load_weights_from_txt<gru_recurrent_bias_t, 210>(br2, "br2.txt");
        nnet::load_weights_from_txt<gru_1_weight_t, 14700>(w3, "w3.txt");
        nnet::load_weights_from_txt<gru_1_recurrent_weight_t, 14700>(wr3, "wr3.txt");
        nnet::load_weights_from_txt<gru_1_bias_t, 210>(b3, "b3.txt");
        nnet::load_weights_from_txt<gru_1_recurrent_bias_t, 210>(br3, "br3.txt");
        nnet::load_weights_from_txt<gru_2_weight_t, 14700>(w4, "w4.txt");
        nnet::load_weights_from_txt<gru_2_recurrent_weight_t, 14700>(wr4, "wr4.txt");
        nnet::load_weights_from_txt<gru_2_bias_t, 210>(b4, "b4.txt");
        nnet::load_weights_from_txt<gru_2_recurrent_bias_t, 210>(br4, "br4.txt");
        nnet::load_weights_from_txt<gru_3_weight_t, 14700>(w5, "w5.txt");
        nnet::load_weights_from_txt<gru_3_recurrent_weight_t, 14700>(wr5, "wr5.txt");
        nnet::load_weights_from_txt<gru_3_bias_t, 210>(b5, "b5.txt");
        nnet::load_weights_from_txt<gru_3_recurrent_bias_t, 210>(br5, "br5.txt");
        nnet::load_weights_from_txt<logit_weight_t, 70>(w6, "w6.txt");
        nnet::load_weights_from_txt<logit_bias_t, 1>(b6, "b6.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[8*70];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    layer3_t layer3_out[8*70];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0

    layer4_t layer4_out[8*70];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0

    layer5_t layer5_out[70];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    nnet::gru_stack<input_t, layer2_t, config2>(syndrome_sequence, layer2_out, w2, wr2, b2, br2); // gru

    nnet::gru_stack<layer2_t, layer3_t, config3>(layer2_out, layer3_out, w3, wr3, b3, br3); // gru_1

    nnet::gru_stack<layer3_t, layer4_t, config4>(layer3_out, layer4_out, w4, wr4, b4, br4); // gru_2

    nnet::gru_stack<layer4_t, layer5_t, config5>(layer4_out, layer5_out, w5, wr5, b5, br5); // gru_3

    nnet::dense<layer5_t, result_t, config6>(layer5_out, layer6_out, w6, b6); // logit

}

