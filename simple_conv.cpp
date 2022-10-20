//Input: 9,9,32
//Number of filters: 43
//Kernel size: 1

#include "simple_conv.h"

#define DMA_WIDTH 64
#define FM_MEM_SIZE 3483

typedef ap_int<4> quant_t; //ver se weights sao signed ou unsigned
typedef hls::axis<quant_t, 0, 0, 0> strmio_t; //para já vamos assumir que stream é de 4 bits


void simple_conv(hls::stream<strmio_t> &strm_in, hls::stream<strmio_t> &strm_out) {
	#pragma HLS INTERFACE ap_ctrl_none port=return
	#pragma HLS INTERFACE axis port=strm_in
	#pragma HLS INTERFACE axis port=strm_out


	strmio_t tmpin, tmpout;
	static quant_t feature_map[FM_MEM_SIZE], weights[WEIGHTS_MEM_SIZE];
}
