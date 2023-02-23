#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

#include "layers_params.h"

#define ARRAYS

#define DMA_WIDTH 64
#define WEIGHTS_WIDTH 4
#define ACT_WIDTH 4
#define NLAYERS 2
#define NPATCHES 6


//typedef ap_fixed<4,2,AP_RND> quant_t;
//typedef ap_fixed<8,4,AP_RND> quant_mult;
//typedef ap_fixed<9,4,AP_RND> quant_accum;

typedef ap_int<4> quant_t;
typedef ap_int<5> quant_sum;
typedef ap_int<7> quant_mult;
typedef ap_int<16> quant_accum;
typedef ap_int<32> quant_reshp;
#define RESHP_FACTOR 8

//typedef float quant_t;
typedef ap_uint<16> count_t;
typedef ap_uint<17> widx_t;
typedef short params_t;
typedef hls::axis<quant_t, 0, 0, 0> strmio_t;


void simple_conv(hls::stream<strmio_t> &strm_in, hls::stream<strmio_t> &strm_out);

//void read_stream(hls::stream<strmio_t> &strm_in, quant_t *weights_l1, quant_t *weights_l2);

#ifdef ARRAYS

template<params_t fm_width, params_t fm_height, params_t output_width, params_t output_height, params_t nbands, params_t kernel_size>
void average_pool(quant_t in_feature_map[fm_width*fm_height*nbands], quant_t out_feature_map[output_height*output_width*nbands]);

template<params_t fm_width, params_t fm_height, params_t nbands_conv, params_t nbands_shortcut>
void add_shortcut(quant_reshp conv_feature_map[fm_width*fm_height*nbands_conv/RESHP_FACTOR], quant_t shortcut[fm_width*fm_height*nbands_shortcut], quant_reshp out_feature_map[fm_width*fm_height*nbands_conv/RESHP_FACTOR]);

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, quant_reshp *weights, params_t PE, bool relu>
void conv_layer_k1(quant_reshp in_feature_map[fm_height*fm_width*nbands/RESHP_FACTOR], quant_reshp out_feature_map[fm_height*fm_width*nfilters/RESHP_FACTOR]);

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, quant_reshp *weights, params_t PE>
void conv_layer_k1_relu(quant_reshp in_feature_map[fm_height*fm_width*nbands/RESHP_FACTOR], quant_reshp out_feature_map[fm_height*fm_width*nfilters/RESHP_FACTOR]);

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, quant_reshp *weights>
void conv_layer_k1_1PE(quant_reshp in_feature_map[fm_height*fm_width*nbands], quant_reshp out_feature_map[fm_height*fm_width*nfilters]);

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, quant_reshp *weights, params_t PE>
void conv_layer_k1_b4k2(quant_reshp in_feature_map[fm_height*fm_width*nbands/RESHP_FACTOR], quant_reshp out_feature_map[2][(fm_height-1)*(fm_width-1)*nfilters/2/RESHP_FACTOR]);

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t output_dim,quant_reshp *weights, params_t PE>
void conv_layer_k2(quant_reshp in_feature_map[2][fm_height*fm_width*nbands/2/RESHP_FACTOR], quant_reshp out_feature_map[(fm_height/2)*(fm_width/2)*nfilters/RESHP_FACTOR]);

void dataflow_func(hls::stream<strmio_t> &strm_in, hls::stream<strmio_t> &strm_out);
void read_ifm(hls::stream<strmio_t> &strm_in, quant_reshp in_feature_map[X1*Y1*Z1], quant_t shortcut_ifm[X1*Y1*Z1]);
void write_ofm(quant_reshp *ofm, hls::stream<strmio_t> &strm_out);

#else

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, quant_t *weights>
void conv_layer_k1(hls::stream<quant_t> &strm_in, hls::stream<quant_t> &strm_out);

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t output_dim,quant_t *weights>
void conv_layer_k2(hls::stream<quant_t> &strm_in, hls::stream<quant_t> &strm_out);

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, quant_t *weights>
void conv_layer_relu(hls::stream<quant_t> &strm_in, hls::stream<quant_t> &strm_out);

void dataflow_func(hls::stream<strmio_t> &strm_in, hls::stream<strmio_t> &strm_out);
void read_ifm(hls::stream<strmio_t> &strm_in, hls::stream<quant_t> &in_feature_map);
void write_ofm(hls::stream<quant_t> &ofm, hls::stream<strmio_t> &strm_out, count_t n_pixels);

#endif

