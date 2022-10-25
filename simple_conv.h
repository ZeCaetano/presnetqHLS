#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

#define DMA_WIDTH 64
#define OUT_FM_MEM_SIZE (FM_WIDTH*FM_HEIGHT*N_FILTERS)
#define INPUT_MEM_SIZE (FM_HEIGHT*FM_WIDTH*N_BANDS)
#define WEIGHTS_MEM_SIZE (N_FILTERS*N_BANDS)

//Argumentos que variam nos templates
#define N_FILTERS 43
#define KERNEL_SIZE 1
#define FM_WIDTH 9
#define FM_HEIGHT 9
#define N_BANDS 32

//typedef ap_int<4> quant_t;
typedef float quant_t;
typedef ap_uint<12> count_t;
typedef short params_t;
typedef hls::axis<float, 0, 0, 0> strmio_t;

void simple_conv(hls::stream<strmio_t> &strm_in, hls::stream<strmio_t> &strm_out);

template<params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t kernel_size>
void layer(hls::stream<strmio_t> &strm_in, hls::stream<strmio_t> &strm_out);
