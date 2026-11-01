#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_recurrent.h"

// hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/wr2.h"
#include "weights/b2.h"
#include "weights/br2.h"
#include "weights/w3.h"
#include "weights/wr3.h"
#include "weights/b3.h"
#include "weights/br3.h"
#include "weights/w4.h"
#include "weights/wr4.h"
#include "weights/b4.h"
#include "weights/br4.h"
#include "weights/w5.h"
#include "weights/wr5.h"
#include "weights/b5.h"
#include "weights/br5.h"
#include "weights/w6.h"
#include "weights/b6.h"


// hls-fpga-machine-learning insert layer-config
// gru
struct config2_1 : nnet::dense_config {
    static const unsigned n_in = 48;
    static const unsigned n_out = 210;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 70;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 10080;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef gru_accum_t accum_t;
    typedef gru_bias_t bias_t;
    typedef gru_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseResource_rf_gt_nin_rem0<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config2_2 : nnet::dense_config {
    static const unsigned n_in = 70;
    static const unsigned n_out = 210;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 70;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 14700;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef gru_accum_t accum_t;
    typedef gru_recurrent_bias_t bias_t;
    typedef gru_recurrent_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseResource_rf_leq_nin<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct sigmoid_config2_recr : nnet::activ_config {
    static const unsigned n_in = 140;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 70;
    typedef gru_table_t table_t;
};

struct tanh_config2 : nnet::activ_config {
    static const unsigned n_in = 70;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 70;
    typedef gru_table_t table_t;
};

struct config2 : nnet::gru_config {
    typedef gru_accum_t accum_t;
    typedef gru_weight_t weight_t;  // Matrix
    typedef gru_recurrent_weight_t recurrent_weight_t;  // Matrix
    typedef gru_bias_t bias_t;  // Vector
    typedef gru_recurrent_bias_t recurrent_bias_t;  // Vector
    typedef config2_1 mult_config1;
    typedef config2_2 mult_config2;
    typedef sigmoid_config2_recr ACT_CONFIG_GRU;
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::sigmoid<x_T, y_T, config_T>;
    typedef tanh_config2 ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::tanh<x_T, y_T, config_T>;
    static const unsigned n_in  = 48;
    static const unsigned n_out = 70;
    static const unsigned n_state = 70;
    static const unsigned n_sequence = 8;
    static const unsigned n_sequence_out = 8;
    static const unsigned io_type = nnet::resource;
    static const unsigned reuse_factor = 70;
    static const bool store_weights_in_bram = false;
    static const bool use_static = true;
    static const bool pytorch_order = false;
    static const bool reset_after = false;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_unit_par = 1;
    static const unsigned n_cycles = 1;
    static const unsigned n_lanes_in = 48;
    static const unsigned n_lanes_state = 70;
    static const unsigned n_weight = 10080;
    static const unsigned n_recurrent_weight = 14700;
};

// gru_1
struct config3_1 : nnet::dense_config {
    static const unsigned n_in = 70;
    static const unsigned n_out = 210;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 70;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 14700;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef gru_1_accum_t accum_t;
    typedef gru_1_bias_t bias_t;
    typedef gru_1_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseResource_rf_leq_nin<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config3_2 : nnet::dense_config {
    static const unsigned n_in = 70;
    static const unsigned n_out = 210;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 70;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 14700;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef gru_1_accum_t accum_t;
    typedef gru_1_recurrent_bias_t bias_t;
    typedef gru_1_recurrent_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseResource_rf_leq_nin<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct sigmoid_config3_recr : nnet::activ_config {
    static const unsigned n_in = 140;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 70;
    typedef gru_1_table_t table_t;
};

struct tanh_config3 : nnet::activ_config {
    static const unsigned n_in = 70;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 70;
    typedef gru_1_table_t table_t;
};

struct config3 : nnet::gru_config {
    typedef gru_1_accum_t accum_t;
    typedef gru_1_weight_t weight_t;  // Matrix
    typedef gru_1_recurrent_weight_t recurrent_weight_t;  // Matrix
    typedef gru_1_bias_t bias_t;  // Vector
    typedef gru_1_recurrent_bias_t recurrent_bias_t;  // Vector
    typedef config3_1 mult_config1;
    typedef config3_2 mult_config2;
    typedef sigmoid_config3_recr ACT_CONFIG_GRU;
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::sigmoid<x_T, y_T, config_T>;
    typedef tanh_config3 ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::tanh<x_T, y_T, config_T>;
    static const unsigned n_in  = 70;
    static const unsigned n_out = 70;
    static const unsigned n_state = 70;
    static const unsigned n_sequence = 8;
    static const unsigned n_sequence_out = 8;
    static const unsigned io_type = nnet::resource;
    static const unsigned reuse_factor = 70;
    static const bool store_weights_in_bram = false;
    static const bool use_static = true;
    static const bool pytorch_order = false;
    static const bool reset_after = false;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_unit_par = 1;
    static const unsigned n_cycles = 1;
    static const unsigned n_lanes_in = 70;
    static const unsigned n_lanes_state = 70;
    static const unsigned n_weight = 14700;
    static const unsigned n_recurrent_weight = 14700;
};

// gru_2
struct config4_1 : nnet::dense_config {
    static const unsigned n_in = 70;
    static const unsigned n_out = 210;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 70;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 14700;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef gru_2_accum_t accum_t;
    typedef gru_2_bias_t bias_t;
    typedef gru_2_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseResource_rf_leq_nin<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config4_2 : nnet::dense_config {
    static const unsigned n_in = 70;
    static const unsigned n_out = 210;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 70;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 14700;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef gru_2_accum_t accum_t;
    typedef gru_2_recurrent_bias_t bias_t;
    typedef gru_2_recurrent_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseResource_rf_leq_nin<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct sigmoid_config4_recr : nnet::activ_config {
    static const unsigned n_in = 140;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 70;
    typedef gru_2_table_t table_t;
};

struct tanh_config4 : nnet::activ_config {
    static const unsigned n_in = 70;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 70;
    typedef gru_2_table_t table_t;
};

struct config4 : nnet::gru_config {
    typedef gru_2_accum_t accum_t;
    typedef gru_2_weight_t weight_t;  // Matrix
    typedef gru_2_recurrent_weight_t recurrent_weight_t;  // Matrix
    typedef gru_2_bias_t bias_t;  // Vector
    typedef gru_2_recurrent_bias_t recurrent_bias_t;  // Vector
    typedef config4_1 mult_config1;
    typedef config4_2 mult_config2;
    typedef sigmoid_config4_recr ACT_CONFIG_GRU;
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::sigmoid<x_T, y_T, config_T>;
    typedef tanh_config4 ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::tanh<x_T, y_T, config_T>;
    static const unsigned n_in  = 70;
    static const unsigned n_out = 70;
    static const unsigned n_state = 70;
    static const unsigned n_sequence = 8;
    static const unsigned n_sequence_out = 8;
    static const unsigned io_type = nnet::resource;
    static const unsigned reuse_factor = 70;
    static const bool store_weights_in_bram = false;
    static const bool use_static = true;
    static const bool pytorch_order = false;
    static const bool reset_after = false;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_unit_par = 1;
    static const unsigned n_cycles = 1;
    static const unsigned n_lanes_in = 70;
    static const unsigned n_lanes_state = 70;
    static const unsigned n_weight = 14700;
    static const unsigned n_recurrent_weight = 14700;
};

// gru_3
struct config5_1 : nnet::dense_config {
    static const unsigned n_in = 70;
    static const unsigned n_out = 210;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 70;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 14700;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef gru_3_accum_t accum_t;
    typedef gru_3_bias_t bias_t;
    typedef gru_3_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseResource_rf_leq_nin<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config5_2 : nnet::dense_config {
    static const unsigned n_in = 70;
    static const unsigned n_out = 210;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 70;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 14700;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef gru_3_accum_t accum_t;
    typedef gru_3_recurrent_bias_t bias_t;
    typedef gru_3_recurrent_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseResource_rf_leq_nin<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct sigmoid_config5_recr : nnet::activ_config {
    static const unsigned n_in = 140;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 70;
    typedef gru_3_table_t table_t;
};

struct tanh_config5 : nnet::activ_config {
    static const unsigned n_in = 70;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 70;
    typedef gru_3_table_t table_t;
};

struct config5 : nnet::gru_config {
    typedef gru_3_accum_t accum_t;
    typedef gru_3_weight_t weight_t;  // Matrix
    typedef gru_3_recurrent_weight_t recurrent_weight_t;  // Matrix
    typedef gru_3_bias_t bias_t;  // Vector
    typedef gru_3_recurrent_bias_t recurrent_bias_t;  // Vector
    typedef config5_1 mult_config1;
    typedef config5_2 mult_config2;
    typedef sigmoid_config5_recr ACT_CONFIG_GRU;
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::sigmoid<x_T, y_T, config_T>;
    typedef tanh_config5 ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::tanh<x_T, y_T, config_T>;
    static const unsigned n_in  = 70;
    static const unsigned n_out = 70;
    static const unsigned n_state = 70;
    static const unsigned n_sequence = 8;
    static const unsigned n_sequence_out = 1;
    static const unsigned io_type = nnet::resource;
    static const unsigned reuse_factor = 70;
    static const bool store_weights_in_bram = false;
    static const bool use_static = true;
    static const bool pytorch_order = false;
    static const bool reset_after = false;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_unit_par = 1;
    static const unsigned n_cycles = 1;
    static const unsigned n_lanes_in = 70;
    static const unsigned n_lanes_state = 70;
    static const unsigned n_weight = 14700;
    static const unsigned n_recurrent_weight = 14700;
};

// logit
struct config6 : nnet::dense_config {
    static const unsigned n_in = 70;
    static const unsigned n_out = 1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 70;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 70;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef logit_accum_t accum_t;
    typedef logit_bias_t bias_t;
    typedef logit_weight_t weight_t;
    typedef layer6_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseResource_rf_leq_nin<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



#endif
