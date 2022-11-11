//-------LAYER 1--------//
//Input: 9,9,32
//Number of filters: 43
//Kernel size: 1
//-------LAYER 2--------//
//Input: 9,9,43
//Number of filters: 43
//Kernel size: 1


#include "simple_conv.h"


quant_t weights_l1[WEIGHTS1], weights_l2[WEIGHTS2];

void simple_conv(hls::stream<strmio_t> &strm_in, hls::stream<strmio_t> &strm_out) {

#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=strm_in
#pragma HLS INTERFACE axis port=strm_out

#pragma HLS DATAFLOW


#ifdef ARRAYS
	quant_t in_feature_map[X1*Y1*Z1], m_feature_map[X2*Y2*Z2], out_feature_map[X2*Y2*NF2];


	read_stream(strm_in, in_feature_map, X1*Y1*Z1);
	layer<0,X1,Y1,Z1,NF1,K1,0> (in_feature_map, m_feature_map, weights_l1);
	layer<1,X2,Y2,Z2,NF2,K2,K1*K1*NF1*Z1> (m_feature_map, out_feature_map, weights_l2);
	write_ofm(out_feature_map, strm_out, X2*Y2*NF2);
#else
	hls::stream<quant_t> m0, m1, m2;

#pragma HLS STREAM variable=m0
#pragma HLS STREAM variable=m1
#pragma HLS STREAM variable=m2
	read_stream(strm_in, m0, X1*Y1*Z1);
	layer<0,X1,Y1,Z1,NF1,K1,0> (m0, m1, weights_l1);
	layer<1,X2,Y2,Z2,NF2,K2,K1*K1*NF1*Z1> (m1, m2, weights_l2);
	write_ofm(m2, strm_out, X2*Y2*NF2);
#endif
}

#ifdef ARRAYS
void read_stream(hls::stream<strmio_t> &strm_in, quant_t *ifm, count_t n_pixels) {
#else
void read_stream(hls::stream<strmio_t> &strm_in, hls::stream<quant_t> &ifm, count_t n_pixels) {
#endif

	strmio_t tmpin;
	quant_t tmpout;

	//Read weights for layer 1
	for(int i = 0; i < WEIGHTS1; i++) {
		tmpin = strm_in.read();
//		printf("Receive %f\n", (float)tmpin.data);
		weights_l1[i] = tmpin.data;
//		if(layer_id == 1) printf("%d  %f-%d\n",i, weights[i], tmpin.last);
		if(tmpin.last == 1) break;
	}

	//Read weights for layer 2
	for(int i = 0; i < WEIGHTS2; i++) {
		tmpin = strm_in.read();
//		printf("Receive %f\n", (float)tmpin.data);
		weights_l2[i] = tmpin.data;
//		if(layer_id == 1) printf("%d  %f-%d\n",i, weights[i], tmpin.last);
		if(tmpin.last == 1) break;
	}
#ifdef ARRAYS
	//Read input fm
	for(count_t i = 0; i < n_pixels; i++) {
		tmpin = strm_in.read();
		ifm[i] = tmpin.data;
//		if(layer_id == 1) printf("%f-%d\n", tmpin.data, i);
		if(tmpin.last == 1) break;
	}
#else
	for(count_t i = 0; i < n_pixels; i++) {
		tmpin = strm_in.read();
		tmpout = tmpin.data;
		ifm.write(tmpout);
		if(tmpin.last == 1) break;
	}
#endif
}

#ifdef ARRAYS
void write_ofm(quant_t *ofm, hls::stream<strmio_t> &strm_out, count_t n_pixels) {
#else
void write_ofm(hls::stream<quant_t> &ofm, hls::stream<strmio_t> &strm_out, count_t n_pixels) {
#endif
	strmio_t tmpout;
#ifndef ARRAYS
	quant_t tmpin;
#endif

	//Write output fm to stream
	for(count_t i = 0; i < n_pixels; i++){
		if(i == n_pixels - 1) tmpout.last = 1;
		else tmpout.last = 0;
#ifdef ARRAYS
		tmpout.data = ofm[i];
#else
		tmpin = ofm.read();
		tmpout.data = tmpin;
#endif
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		strm_out.write(tmpout);
	}
}

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t kernel_size, params_t weights_start>
#ifdef ARRAYS
void layer(quant_t *in_feature_map, quant_t *out_feature_map, quant_t *weights) {
#else
void layer(hls::stream<quant_t> &strm_in, hls::stream<quant_t> &strm_out, quant_t *weights) {
#endif

#ifndef ARRAYS
	quant_t tmpin, tmpout;
	quant_t pixel;
#endif

	//Convolution
	quant_mult acc = 0;
	quant_accum acc_arr[nfilters];
	widx_t kernel_idx = 0;

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
							acc += weights[kernel_idx] * in_feature_map[input_idx];
						}
					}
				}
				out_feature_map[z*fm_width*fm_height + i*fm_height + j] = (quant_t)acc;
//				if(j == fm_width -1 && i == fm_height - 1 && z == nfilters - 1) tmpout.last = 1;
//				else tmpout.last = 0;
//				tmpout.data = acc;
//				tmpout.keep = 0xF;
//				tmpout.strb = 0xF;
//				strm_out.write(tmpout);
			}
		}
	}


//	loop_inputx:
//	for(count_t i = 0; i < fm_width; i++) {
//		loop_inputy:
//		for(count_t j = 0; j < fm_height; j++) {
//			kernel_idx = 0;
//			loop_bands:
//			for(count_t k = 0; k < nbands; k++) {
//#ifndef ARRAYS
//				tmpin = strm_in.read();
//				pixel = tmpin;
//#endif
//				loop_filters:
//				for(count_t z = 0; z < nfilters; z++) {
//					acc = 0;
//					loop_kernelx:
//					for(count_t x = 0; x < kernel_size; x++) {
//						loop_kernely:
//						for(count_t y = 0; y < kernel_size; y++) {
//#pragma HLS PIPELINE
////							/* Kernel index */
////							count_t kernel_idx =
////									(z*kernel_size*kernel_size*nbands) +                   /* nfilter */
////									(k * (kernel_size * kernel_size) + 		               /* band*/
////									(x * kernel_size +                                     /* kernel row */
////									y));                                                   /* kernel column */
//
//#ifdef ARRAYS
//							/* Input matrix index */
//							count_t input_idx =
//									k * (fm_width * fm_height) + ((i + x) * fm_height +  /* input row */
//									j + y);                                      /* input column */
//#endif
////							printf("%d IP - %d %d %f\n",z, k, (x*kernel_size) + y, (float)weights[kernel_idx]);
//#ifdef ARRAYS
//							acc += weights[kernel_idx] * in_feature_map[input_idx];
//#else
//							acc += weights[kernel_idx] *  pixel ;
//#endif
//							kernel_idx++;
//						}
//					}
//					if(k == 0)
//						acc_arr[z] = acc;
//					else
//						acc_arr[z] += acc;
//					if(k == nbands-1) {  //If it's the last band, put the accum in the output feature map and reset the accum array
//#ifdef ARRAYS
//						out_feature_map[z*fm_width*fm_height + i*fm_height + j] = (quant_t)acc_arr[z];
//#else
////						printf("sending pixel number %d of filter %d \n", (i*fm_height)+j, z);
//						tmpout = (quant_t)acc_arr[z];
//						strm_out.write(tmpout);
//#endif
//						acc_arr[z] = 0;
//					}
//
//				}
//			}
//		}
//	}

//#ifndef ARRAYS
//	if(layer_id < NLAYERS-1){  //only send weights if it's not the last layer
//		printf("Layer %d is sending weights again\n", layer_id);
//		for (int i = 0 ; i < WEIGHTS_MEM_SIZE; i++) {
//			if(i == WEIGHTS_MEM_SIZE - 1) tmpout.last = 1;
//			else tmpout.last = 0;
////			printf("weights sent inside ip %f  - %d\n", weights[i], i);
//			tmpout.data = weights[i];
//			tmpout.keep = 0xF;
//			tmpout.strb = 0xF;
//			strm_out.write(tmpout);
//		}
//	}
//#endif
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
//		for(int k = 0; k < nfilters; k++) {
//			for (int x = 0; x < fm_height; x++) {
//				for (int j = 0; j < fm_width; j++) {
//					printf("%f ", out_feature_map[(k+1)*x * fm_width + j]);
//				}
//				printf("\n\r");
//			}
//		}
//	}
	return;
}
