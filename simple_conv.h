#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

#define ARRAYS

#define DMA_WIDTH 64
#define INPUT1_MEM_SIZE (X1*Y1*Z1)
#define OUT1_FM_MEM_SIZE (X1*Y1*NF1)
#define INPUT2_MEM_SIZE (X2*Y2*Z2)
#define OUT2_FM_MEM_SIZE (X2*Y2*NF2)
#define WEIGHTS_MEM_SIZE (LAYER1_WEIGHTS+LAYER2_WEIGHTS)
#define LAYER1_WEIGHTS (NF1*Z1*K1*K1)
#define LAYER2_WEIGHTS (NF2*Z2*K2*K2)
#define NLAYERS 2

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
#define WEIGHTS1 (Z1*NF1*K1*K1)
//-------LAYER 2--------//
//Input: 9,9,43
//Number of filters: 43
//Kernel size: 1
#define X2 9
#define Y2 9
#define Z2 43
#define NF2 43
#define K2 1
#define WEIGHTS2 (Z2*NF2*K2*K2)

typedef ap_fixed<4,2,AP_RND> quant_t;
typedef ap_fixed<8,4,AP_RND> quant_mult;
typedef ap_fixed<9,4,AP_RND> quant_accum;
//typedef float quant_t;
typedef ap_uint<12> count_t;
typedef short params_t;
typedef hls::axis<quant_t, 0, 0, 0> strmio_t;


void simple_conv(hls::stream<strmio_t> &strm_in, hls::stream<strmio_t> &strm_out);

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t kernel_size, params_t weights_start>
#ifdef ARRAYS
void layer(quant_t *in_feature_map, quant_t *out_feature_map, quant_t *weights);
#else
void layer(hls::stream<quant_t> &strm_in, hls::stream<quant_t> &strm_out, quant_t *weights);
#endif

#ifdef ARRAYS
void read_stream(hls::stream<strmio_t> &strm_in, quant_t *ifm, count_t n_pixels);
void write_ofm(quant_t *ofm, hls::stream<strmio_t> &strm_out, count_t n_pixels);
#else
void read_stream(hls::stream<strmio_t> &strm_in, hls::stream<quant_t> &ifm, count_t n_pixels);
void write_ofm(hls::stream<quant_t> &ofm, hls::stream<strmio_t> &strm_out, count_t n_pixels);
#endif

