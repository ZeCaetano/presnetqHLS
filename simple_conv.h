#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

#include "layers_params.h"

#define ARRAYS

#define WEIGHTS_WIDTH 2
#define ACT_WIDTH 4
#define NPATCHES 6
#define NCLASSES 16

typedef ap_int<4> quant_act;
typedef ap_uint<4> quant_uact;
typedef ap_int<2> quant_wght;
typedef ap_int<8> quant_bias;


typedef ap_int<5> quant_sum;
typedef ap_int<7> quant_mult;
typedef ap_int<21> quant_accum;

typedef ap_uint<32> quant_reshp;
typedef ap_uint<16> wght_reshp;
typedef ap_uint<32> act_reshp;
#define RESHP_FACTOR 8

typedef ap_fixed<32, 32-(SFEDS3+SFEW_FC)> data_out;
typedef ap_uint<32> data_in;
typedef short params_t;
typedef hls::axis<data_in, 0, 0, 0> strmi_t;
typedef hls::axis<data_out, 0, 0, 0> strmo_t;


void simple_conv(hls::stream<strmi_t> &strm_in, hls::stream<strmo_t> &strm_out);

//void read_stream(hls::stream<strmio_t> &strm_in, quant_act *weights_l1, quant_t *weights_l2);


template<params_t fm_width, params_t fm_height, params_t output_width, params_t output_height, params_t nbands, params_t kernel_size, params_t sfe_i, params_t sfe_o>
void average_pool(act_reshp in_feature_map[fm_width*fm_height*nbands/RESHP_FACTOR], act_reshp out_feature_map[output_height*output_width*nbands/RESHP_FACTOR]);

template<params_t fm_width, params_t fm_height, params_t output_width, params_t output_height, params_t nbands, params_t kernel_size, params_t sfe_i, params_t sfe_o>
void average_pool_unsigned(act_reshp in_feature_map[fm_width*fm_height*nbands/RESHP_FACTOR], act_reshp out_feature_map[output_height*output_width*nbands/RESHP_FACTOR]);

template<params_t fm_width, params_t fm_height, params_t nbands_conv, params_t nbands_shortcut, params_t sfe_conv, params_t sfe_shortcut, params_t sfe_o>
void add_shortcut(act_reshp conv_feature_map[fm_width*fm_height*nbands_conv/RESHP_FACTOR], act_reshp shortcut[fm_width*fm_height*nbands_shortcut/RESHP_FACTOR], act_reshp out_feature_map[fm_width*fm_height*nbands_conv/RESHP_FACTOR]);

template<params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t sfe_i, params_t sfe_weights, params_t sfe_o, wght_reshp *weights, params_t PE, bool relu>
void conv_layer_k1(act_reshp in_feature_map[fm_height*fm_width*nbands/RESHP_FACTOR], act_reshp out_feature_map[fm_height*fm_width*nfilters/RESHP_FACTOR]);

template<params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t sfe_i, params_t sfe_weights, params_t sfe_o, wght_reshp *weights, params_t PE, bool relu>
void conv_layer_k1_unsigned(act_reshp in_feature_map[fm_height*fm_width*nbands/RESHP_FACTOR], act_reshp out_feature_map[fm_height*fm_width*nfilters/RESHP_FACTOR]);

template<params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, wght_reshp *weights, params_t PE>
void conv_layer_k1_relu(act_reshp in_feature_map[fm_height*fm_width*nbands/RESHP_FACTOR], act_reshp out_feature_map[fm_height*fm_width*nfilters/RESHP_FACTOR]);

template<params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, wght_reshp *weights>
void conv_layer_k1_1PE(act_reshp in_feature_map[fm_height*fm_width*nbands], act_reshp out_feature_map[fm_height*fm_width*nfilters]);

template<params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t sfe_i, params_t sfe_weights, params_t sfe_o, wght_reshp *weights, params_t PE,bool relu>
void conv_layer_k1_b4k2_x9(act_reshp in_feature_map[fm_height*fm_width*nbands/RESHP_FACTOR], act_reshp out_feature_map[2][(fm_height-1)*(fm_width-1)*nfilters/2/RESHP_FACTOR]);

template<params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t sfe_i, params_t sfe_weights, params_t sfe_o, wght_reshp *weights, params_t PE,bool relu>
void conv_layer_k1_b4k2_x4(act_reshp in_feature_map[fm_height*fm_width*nbands/RESHP_FACTOR], act_reshp out_feature_map[2][(fm_height)*(fm_width)*nfilters/2/RESHP_FACTOR]);

template<params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t output_dim, params_t sfe_i, params_t sfe_weights, params_t sfe_o,wght_reshp *weights, params_t PE, bool relu>
void conv_layer_k2(act_reshp in_feature_map[2][fm_height*fm_width*nbands/2/RESHP_FACTOR], act_reshp out_feature_map[(fm_height/2)*(fm_width/2)*nfilters/RESHP_FACTOR]);

template<params_t input_size, params_t nfilters, wght_reshp *weights, quant_bias *bias, params_t sfe_i, params_t sfe_bias, params_t sfe_weights, params_t sfe_o>
void fully_connected(act_reshp input_fm[input_size/RESHP_FACTOR], data_out output_fm[nfilters]);

void read_ifm(hls::stream<strmi_t> &strm_in, act_reshp in_feature_map[X1*Y1*Z1/RESHP_FACTOR]);
void write_ofm(data_out *ofm, hls::stream<strmo_t> &strm_out);

template<params_t width, params_t height, params_t nbands>
void gen_shortcut(act_reshp fm[width*height*nbands/RESHP_FACTOR], act_reshp fm1[width*height*nbands/RESHP_FACTOR], act_reshp fm2[width*height*nbands/RESHP_FACTOR]);

template<params_t width, params_t height, params_t nbands, params_t sfe_i,params_t sfe_o>
void relu(act_reshp in[width*height*nbands/RESHP_FACTOR], act_reshp out[width*height*nbands/RESHP_FACTOR]);
