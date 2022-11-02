//-------LAYER 1--------//
//Input: 9,9,32
//Number of filters: 43
//Kernel size: 1
//-------LAYER 2--------//
//Input: 9,9,43
//Number of filters: 43
//Kernel size: 1


#include "simple_conv.h"

#define CONST

#ifdef CONST
static quant_t weights[WEIGHTS_MEM_SIZE];
#endif

void simple_conv(hls::stream<strmio_t> &strm_in, hls::stream<strmio_t> &strm_out) {

	#pragma HLS INTERFACE ap_ctrl_none port=return
	#pragma HLS INTERFACE axis port=strm_in
	#pragma HLS INTERFACE axis port=strm_out

	hls::stream<strmio_t> m1;

	#pragma HLS stream variable=m1 type=fifo

#ifdef CONST
	read_weights(strm_in, weights);
#endif

	#pragma HLS DATAFLOW
	layer<0,X1,Y1,Z1,NF1,K1,0> (strm_in, m1);
	layer<1,X2,Y2,Z2,NF2,K2,K1*K1*NF1*Z1> (m1, strm_out);
}


void read_weights(hls::stream<strmio_t> &strm_in, quant_t *weights) {
	strmio_t tmpin;

	//Read weights
	for(int i = 0; i < WEIGHTS_MEM_SIZE; i++) {
		tmpin = strm_in.read();
		weights[i] = tmpin.data;
//		if(layer_id == 1) printf("%d  %f-%d\n",i, weights[i], tmpin.last);
		if(tmpin.last == 1) break;
	}
}

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t kernel_size, params_t weights_start>
void layer(hls::stream<strmio_t> &strm_in, hls::stream<strmio_t> &strm_out) {


	strmio_t tmpin, tmpout;
	static quant_t in_feature_map[fm_width*fm_height*nbands], out_feature_map[fm_width*fm_height*nfilters];

#ifndef CONST
	static quant_t weights[WEIGHTS_MEM_SIZE];
	read_weights(strm_in, weights);
#endif

	//Read input fm
	for(count_t i = 0; i < fm_width*fm_height*nbands; i++) {
		tmpin = strm_in.read();
		in_feature_map[i] = tmpin.data;
//		if(layer_id == 1) printf("%f-%d\n", tmpin.data, i);
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
//Fazer leitura px=strm
					loop_kernelx:
					for(count_t x = 0; x < kernel_size; x++) {
						loop_kernely:
						for(count_t y = 0; y < kernel_size; y++) {
#pragma HLS PIPELINE
							/* Kernel index */
							count_t kernel_idx =
									weights_start +                                        /* layer */
									(z*kernel_size*kernel_size*nbands) +                   /* nfilter */
									(k * (kernel_size * kernel_size) + 		               /* band*/
									(x * kernel_size +                                     /* kernel row */
									y));                                                   /* kernel column */


							/* Input matrix index */
							count_t input_idx =
									k * (fm_width * fm_height) + ((i + x) * fm_height +  /* input row */
									j + y);                                      /* input column */
//							if(layer_id == 1) {
//								printf("IP weight %f  - %d\n", weights[kernel_idx], kernel_idx);
//								printf("IP pixel %f  - %d\n",in_feature_map[input_idx], input_idx);
//							}
							//normalize pixel
							acc += weights[kernel_idx] * ((float) in_feature_map[input_idx] / 255 - 0.5F) / 0.5F;
						}
					}
				}
//				out_feature_map[z*fm_width*fm_height + i*fm_height + j] = acc;
				if(j == fm_width -1 && i == fm_height - 1 && z == nfilters - 1) tmpout.last = 1;
				else tmpout.last = 0;
				tmpout.data = acc;
				tmpout.keep = 0xF;
				tmpout.strb = 0xF;
				strm_out.write(tmpout);
			}
		}
	}
#ifndef CONST
	if(layer_id < NLAYERS-1){  //only send weights if it's not the last layer
		printf("Layer %d is sending weights again\n", layer_id);
		for (int i = 0 ; i < WEIGHTS_MEM_SIZE; i++) {
			if(i == WEIGHTS_MEM_SIZE - 1) tmpout.last = 1;
			else tmpout.last = 0;
//			printf("weights sent inside ip %f  - %d\n", weights[i], i);
			tmpout.data = weights[i];
			tmpout.keep = 0xF;
			tmpout.strb = 0xF;
			strm_out.write(tmpout);
		}
	}
#endif
//	for(count_t i = 0; i < fm_width*fm_height*nfilters; i++){
//		if(i == fm_width*fm_height*nfilters - 1) tmpout.last = 1;
//		else tmpout.last = 0;
//		tmpout.data = out_feature_map[i];
//		tmpout.keep = 0xF;
//		tmpout.strb = 0xF;
//		strm_out.write(tmpout);
//	}

//	if(layer_id == 1){
//		printf("HARDWARE Output Image 1\n\r");
//		for(int k = 0; k < N_FILTERS; k++) {
//			for (int x = 0; x < FM_HEIGHT; x++) {
//				for (int j = 0; j < FM_WIDTH; j++) {
//					printf("%f ", out_feature_map[(k+1)*x * FM_WIDTH + j]);
//				}
//				printf("\n\r");
//			}
//		}
//	}
	return;
}
