#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

#define DMA_WIDTH 64
#define FM_MEM_SIZE 3483
#define WEIGHTS_MEM_SIZE 1376

//Argumentos que variam nos templates
#define N_FILTERS 43
#define KERNEL_SIZE 1
#define FM_WIDTH 9
#define FM_HEIGHT 9
#define N_BANDS 32

//typedef ap_int<4> quant_t;
typedef float quant_t;
typedef hls::axis<float, 0, 0, 0> strmio_t;

void simple_conv(hls::stream<strmio_t> &strm_in, hls::stream<strmio_t> &strm_out);
