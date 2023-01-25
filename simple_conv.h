#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

#define ARRAYS

#define DMA_WIDTH 64
#define WEIGHTS_WIDTH 4
#define ACT_WIDTH 4
#define NLAYERS 2
#define NPATCHES 1

//Argumentos que variam nos templates
//-------LAYER 1--------//
//Input: 9,9,32
//Number of filters: 43
//Kernel size: 1
#define X1 9
#define Y1 9
#define Z1 32
#define NF1 43
#define K1 1
#define LAYER1_WEIGHTS (NF1*Z1*K1*K1)
#define INPUT1_MEM_SIZE (X1*Y1*Z1)
#define OUT1_FM_MEM_SIZE (X2*Y2*NF1)
//-------LAYER 2--------//
//Input: 9,9,43
//Number of filters: 43
//Kernel size: 2
#define X2 9
#define Y2 9
#define Z2 43
#define NF2 43
#define K2 2
#define LAYER2_WEIGHTS (NF2*Z2*K2*K2)
#define INPUT2_MEM_SIZE (X2*Y2*Z2)
#define OUT2_FM_MEM_SIZE (X3*Y3*NF2)
#define WEIGHTS_MEM_SIZE (LAYER1_WEIGHTS+LAYER2_WEIGHTS)
//-------LAYER 3--------//
//Input: 4,4,43
//Number of filters: 172
//Kernel size: 2
#define X3 4
#define Y3 4
#define Z3 43
#define NF3 172
#define K3 1
#define LAYER3_WEIGHTS (Z3*NF3*K3*K3)
#define OUTPUT_MEM_SIZE (X3*Y3*Z3)
//-------DOWNSAMPLING LAYER--------//
//Input: LAYER 1
//Output: 4,4,32
//Kernel size: 2
#define XDS 4
#define YDS 4
#define KDS 2
#define OUTDS_FM_MEM_SIZE (XDS*YDS*Z1)
////Average pool test
//#define X1 8
//#define Y1 8
//#define Z1 3
//#define X2 4
//#define Y2 4
//#define Z2 3

//typedef ap_fixed<4,2,AP_RND> quant_t;
//typedef ap_fixed<8,4,AP_RND> quant_mult;
//typedef ap_fixed<9,4,AP_RND> quant_accum;

typedef ap_int<4> quant_t;
typedef ap_int<6> quant_sum;
typedef ap_int<7> quant_mult;
typedef ap_int<16> quant_accum;


//typedef float quant_t;
typedef ap_uint<13> count_t;
typedef ap_uint<17> widx_t;
typedef short params_t;
typedef hls::axis<quant_t, 0, 0, 0> strmio_t;


void simple_conv(hls::stream<strmio_t> &strm_in, hls::stream<strmio_t> &strm_out);

void read_stream(hls::stream<strmio_t> &strm_in, quant_t *weights_l1, quant_t *weights_l2);

#ifdef ARRAYS

template<params_t fm_width, params_t fm_height, params_t output_width, params_t output_height, params_t nbands, params_t kernel_size>
void average_pool(quant_t in_feature_map[fm_width*fm_height*nbands], quant_t out_feature_map[output_height*output_width*nbands]);

template<params_t fm_width, params_t fm_height, params_t nbands_conv, params_t nbands_shortcut>
void add_shortcut(quant_t conv_feature_map[fm_width*fm_height*nbands_conv], quant_t shortcut[fm_width*fm_height*nbands_shortcut], quant_t out_feature_map[fm_width*fm_height*nbands_conv]);

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, quant_t *weights>
void conv_layer_k1(quant_t in_feature_map[fm_height*fm_width*nbands], quant_t out_feature_map[fm_height*fm_width*nfilters]);

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, quant_t *weights>
void conv_layer_k1_b4k2(quant_t in_feature_map[fm_height*fm_width*nbands], quant_t out_feature_map[2][(fm_height-1)*(fm_width-1)*nfilters/2]);

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t output_dim,quant_t *weights>
void conv_layer_k2(quant_t in_feature_map[2][fm_height*fm_width*nbands/2], quant_t out_feature_map[(fm_height/2)*(fm_width/2)*nfilters]);

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, quant_t *weights>
void conv_layer_relu(quant_t in_feature_map[], quant_t out_feature_map[]);

void dataflow_func(hls::stream<strmio_t> &strm_in, hls::stream<strmio_t> &strm_out);
void read_ifm(hls::stream<strmio_t> &strm_in, quant_t in_feature_map[X1*Y1*Z1]);
void write_ofm(quant_t *ofm, hls::stream<strmio_t> &strm_out);

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

