//Input: 9,9,32
//Number of filters: 43
//Kernel size: 1

#include "simple_conv.h"


void simple_conv(hls::stream<strmio_t> &strm_in, hls::stream<strmio_t> &strm_out) {
	layer<FM_WIDTH,FM_HEIGHT,N_BANDS,N_FILTERS,KERNEL_SIZE> (strm_in, strm_out);
}


template<params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t kernel_size>
void layer(hls::stream<strmio_t> &strm_in, hls::stream<strmio_t> &strm_out) {

	#pragma HLS INTERFACE ap_ctrl_none port=return
	#pragma HLS INTERFACE axis port=strm_in
	#pragma HLS INTERFACE axis port=strm_out


	strmio_t tmpin, tmpout;
	static quant_t in_feature_map[INPUT_MEM_SIZE], out_feature_map[OUT_FM_MEM_SIZE], weights[WEIGHTS_MEM_SIZE];
	count_t i;

	//Read weights
	for(i = 0; i < WEIGHTS_MEM_SIZE; i++) {
		tmpin = strm_in.read();
		weights[i] = tmpin.data;
//		printf("%d  %f-%d\n",i, weights[i], tmpin.last);
		if(tmpin.last == 1) break;
	}
	//Read input fm
	for(i = 0; i < INPUT_MEM_SIZE; i++) {
		tmpin = strm_in.read();
		in_feature_map[i] = tmpin.data;
//		printf("%d  %f-%d\n",i, tmpin.data, tmpin.last);
		if(tmpin.last == 1) break;
	}
//	printf("Received all weights and pixels\n");


	//Convolution
	float acc = 0;
	loop_filters:
	for(count_t z = 0; z < nfilters; z++) {
		loop_inputx:
		for(count_t i = 0; i < fm_width; i++) {
			loop_inputy:
			for(count_t j = 0; j < fm_height; j++) {
				acc = 0;
				loop_bands:
				for(count_t k = 0; k < nbands; k++) {
					loop_kernelx:
					for(count_t x = 0; x < kernel_size; x++) {
						loop_kernely:
						for(count_t y = 0; y < kernel_size; y++) {
#pragma HLS PIPELINE
							/* Kernel index */
							count_t kernel_idx =
									(z*kernel_size*kernel_size*nbands) +         /* nfilter */
									(k * (kernel_size * kernel_size) + 		     /* band*/
									(x * kernel_size +                           /* kernel row */
									y));                                         /* kernel column */

//							printf("%d ", kernel_1d_idx);
							/* Input matrix index */
							count_t input_idx =
									k * (fm_width * fm_height) + ((i + x) * fm_height +  /* input row */
									j + y);                                      /* input column */

							//normalize pixel
							acc += weights[kernel_idx] * ((float) in_feature_map[input_idx] / 255 - 0.5F) / 0.5F;
						}
					}
				}
				out_feature_map[z*fm_width*fm_height + i*fm_height + j] = acc;
//				if(j == FM_WIDTH -1 && i == FM_HEIGHT - 1 && z == N_FILTERS - 1) tmpout.last = 1;
//				else tmpout.last = 0;
//				tmpout.data = acc;
//				tmpout.keep = 0xF;
//				tmpout.strb = 0xF;
//				strm_out.write(tmpout);
			}
		}
	}

	for(count_t i = 0; i < OUT_FM_MEM_SIZE; i++){
		if(i == OUT_FM_MEM_SIZE - 1) tmpout.last = 1;
		else tmpout.last = 0;
		tmpout.data = out_feature_map[i];
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		strm_out.write(tmpout);
	}
	return;
}
