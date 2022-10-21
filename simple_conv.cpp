//Input: 9,9,32
//Number of filters: 43
//Kernel size: 1

#include "simple_conv.h"

void simple_conv(hls::stream<strmio_t> &strm_in, hls::stream<strmio_t> &strm_out) {

	#pragma HLS INTERFACE ap_ctrl_none port=return
	#pragma HLS INTERFACE axis port=strm_in
	#pragma HLS INTERFACE axis port=strm_out


	strmio_t tmpin, tmpout;
	static quant_t in_feature_map[FM_MEM_SIZE], out_feature_map[FM_MEM_SIZE], weights[WEIGHTS_MEM_SIZE];
	int i, nfilters = N_FILTERS, kernel_size = KERNEL_SIZE, nbands = N_BANDS, fm_dim = FM_WIDTH;

	//Read weights
	for(i = 0; i < WEIGHTS_MEM_SIZE; i++) {
		tmpin = strm_in.read();
		((float*)weights)[i] = tmpin.data;
		if(tmpin.last == 1) break;
	}

	//Read input fm
	for(i = 0; i < FM_MEM_SIZE; i++) {
		tmpin = strm_in.read();
		((float*)in_feature_map)[i] = tmpin.data;
		if(tmpin.last == 1) break;
	}


	//Convolution
	float acc = 0;

	loop_filters:
	for(int z = 0; z < nfilters; i++) {
		loop_inputx:
		for(int i = 0; i < fm_dim; i++) {
			loop_inputy:
			for(int j = 0; j < fm_dim; i++) {
				acc = 0;
				loop_bands:
				for(int k = 0; k < nbands; i++) {
					loop_kernelx:
					for(int x = 0; x < kernel_size; x++) {
						loop_kernely:
						for(int y = 0; y < kernel_size; y++) {

							/* Kernel index */
							int kernel_idx =
									(z*kernel_size*kernel_size*nbands) +         /* nfilter */
									(k * (kernel_size * kernel_size) + 		     /* band*/
									(x * kernel_size +                           /* kernel row */
									y));                                         /* kernel column */

//							printf("%d ", kernel_1d_idx);
							/* Input matrix index */
							int input_idx =
									k * (fm_dim * fm_dim) + ((i + x) * fm_dim +  /* input row */
									j + y);                                      /* input column */

							//normalize pixel
							acc += weights[kernel_idx] * ((float) in_feature_map[input_idx] / 255 - 0.5F) / 0.5F;
						}
					}
				}
				if(j == FM_WIDTH -1 && i == FM_HEIGHT - 1 && z == N_FILTERS - 1) tmpout.last = 1;
				else tmpout.last = 0;
				tmpout.data = acc;
				tmpout.keep = 0xF;
				tmpout.strb = 0xF;
				strm_out.write(tmpout);
			}
		}
	}
}
