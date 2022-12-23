//-------LAYER 1--------//
//Input: 9,9,32
//Number of filters: 43
//Kernel size: 1
//-------LAYER 2--------//
//Input: 9,9,43
//Number of filters: 43
//Kernel size: 1


#include "simple_conv.h"
#include "weights.h"


void simple_conv(hls::stream<strmio_t> &strm_in, hls::stream<strmio_t> &strm_out) {

#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=strm_in
#pragma HLS INTERFACE axis port=strm_out


#ifdef ARRAYS

//	read_stream(strm_in, weights_l1, weights_l2);

	for(int i = 0; i < NPATCHES; i++){
		dataflow_func(strm_in, strm_out);
	}
#else

//	read_stream(strm_in, weights_l1, weights_l2);

	for(int i = 0; i < NPATCHES; i++){
		dataflow_func(strm_in, strm_out);
	}
#endif
}


void dataflow_func(hls::stream<strmio_t> &strm_in, hls::stream<strmio_t> &strm_out){
#pragma HLS DATAFLOW

#ifdef ARRAYS
	quant_t in_feature_map[X1*Y1*Z1], m_feature_map[X2*Y2*Z2], out_feature_map[X2*Y2*NF2];

	read_ifm(strm_in, in_feature_map);

	conv_layer_k1<0,X1,Y1,Z1,NF1, weights_l1> (in_feature_map, m_feature_map);
	conv_layer_k1<1,X2,Y2,Z2,NF2, weights_l2> (m_feature_map, out_feature_map);

	write_ofm(out_feature_map, strm_out, X2*Y2*NF2);
#else
	hls::stream<quant_t> m0, m1, m2;

	#pragma HLS STREAM variable=m0 depth=2
	#pragma HLS STREAM variable=m1 depth=2
	#pragma HLS STREAM variable=m2 depth=2

	read_ifm(strm_in, m0);

	conv_layer_k1<0,X1,Y1,Z1,NF1, weights_l1> (m0, m1);
	conv_layer_k1<1,X2,Y2,Z2,NF2, weights_l2> (m1, m2);

	write_ofm(m2, strm_out, X2*Y2*NF2);
#endif
}


#ifdef ARRAYS
void read_ifm(hls::stream<strmio_t> &strm_in, quant_t *in_feature_map){
#else
void read_ifm(hls::stream<strmio_t> &strm_in, hls::stream<quant_t> &in_feature_map){
#endif
	strmio_t tmpin;

#ifdef ARRAYS
	//read input fm
	for(count_t i = 0; i < X1*Y1*Z1; i++) {
		tmpin = strm_in.read();
		in_feature_map[i] = tmpin.data;
//		if(layer_id == 1) printf("%f-%d\n", tmpin.data, i);
		if(tmpin.last == 1) break;
	}
#else
	quant_t tmpout;
	for(count_t i = 0; i < X1*Y1*Z1; i++) {
		tmpin = strm_in.read();
		tmpout = tmpin.data;
		in_feature_map.write(tmpout);
		if(tmpin.last == 1) break;
	}
#endif
}

void read_stream(hls::stream<strmio_t> &strm_in, quant_t *weights_l1, quant_t *weights_l2) {

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

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, quant_t *weights>
#ifdef ARRAYS
void conv_layer_k1(quant_t in_feature_map[fm_height*fm_width*nbands], quant_t out_feature_map[fm_height*fm_width*nfilters]) {
#else
void conv_layer_k1(hls::stream<quant_t> &strm_in, hls::stream<quant_t> &strm_out) {
#endif

#ifndef ARRAYS
	quant_t tmpin, tmpout;
	quant_t pixel;
#endif

	//Convolution
	quant_mult acc = 0;
	quant_accum acc_arr[nfilters];
	int kernel_idx = 0;

	loop_inputx:
	for(count_t i = 0; i < fm_width; i++) {
		loop_inputy:
		for(count_t j = 0; j < fm_height; j++) {
			kernel_idx = 0;
			loop_bands:
			for(count_t k = 0; k < nbands; k++) {
				loop_filters:
				for(count_t z = 0; z < nfilters; z++) {
#pragma HLS PIPELINE
#ifndef ARRAYS
					if(z == 0){
						tmpin = strm_in.read();
						pixel = tmpin;
					}
#endif
//					acc = 0;
#ifdef ARRAYS
					/* Input matrix index */
					count_t input_idx =
							k * (fm_width * fm_height) + (i * fm_height +    /* input row */
							j);                                              /* input column */
#endif
//							printf("%d IP - %d %d %f\n",z, k, (x*kernel_size) + y, (float)weights[kernel_idx]);
#ifdef ARRAYS
					acc = weights[kernel_idx] * in_feature_map[input_idx];
#else
					acc = weights[kernel_idx] *  pixel;
#endif
					kernel_idx++;

					if(k == 0)
						acc_arr[z] = acc;
					else
						acc_arr[z] += acc;
					if(k == nbands-1) {  //If it's the last band, put the accum in the output feature map and reset the accum array
#ifdef ARRAYS
						out_feature_map[z*fm_width*fm_height + i*fm_height + j] = (quant_t)acc_arr[z]; //ver o fator de escala aqui
						//aplicar função de ativação aqui
#else
//						printf("sending pixel number %d of filter %d \n", (i*fm_height)+j, z);
						tmpout = (quant_t)acc_arr[z];
						strm_out.write(tmpout);
#endif
						acc_arr[z] = 0;
					}

				}
			}
		}
	}
	return;
}

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, quant_t *weights>
#ifdef ARRAYS
void conv_layer_k2(quant_t in_feature_map[fm_height*fm_width*nbands], quant_t out_feature_map[fm_height*fm_width*nfilters]) {
#else
void conv_layer_k2(hls::stream<quant_t> &strm_in, hls::stream<quant_t> &strm_out) {
#endif

#ifndef ARRAYS
	quant_t tmpin, tmpout;
	quant_t pixel;
#endif
	//Convolution
	quant_mult acc = 0;
	quant_accum acc_arr[nfilters];
	int kernel_size = 2;
	int kernel_idx = 0;

	loop_inputx:
	for(count_t i = 0; i < fm_width; i++) {
		loop_inputy:
		for(count_t j = 0; j < fm_height; j++) {
			kernel_idx = 0;
			loop_bands:
			for(count_t k = 0; k < nbands; k++) {
#ifndef ARRAYS
				tmpin = strm_in.read();
				pixel = tmpin;
#endif
				loop_filters:
				for(count_t z = 0; z < nfilters; z++) {
#pragma HLS PIPELINE
					acc = 0;
					loop_kernelx:
					for(count_t x = 0; x < kernel_size; x++) {
						loop_kernely:
						for(count_t y = 0; y < kernel_size; y++) {

#ifdef ARRAYS
							/* Input matrix index */
							count_t input_idx =
									k * (fm_width * fm_height) + ((i + x) * fm_height +  /* input row */
									j + y);                                      /* input column */
#endif
//							printf("%d IP - %d %d %f\n",z, k, (x*kernel_size) + y, (float)weights[kernel_idx]);
#ifdef ARRAYS
							acc += weights[kernel_idx] * in_feature_map[input_idx];
#else
							acc += weights[kernel_idx] *  pixel;
#endif
							kernel_idx++;
						}
					}
					if(k == 0)
						acc_arr[z] = acc;
					else
						acc_arr[z] += acc;
					if(k == nbands-1) {  //If it's the last band, put the accum in the output feature map and reset the accum array
#ifdef ARRAYS
						out_feature_map[z*fm_width*fm_height + i*fm_height + j] = (quant_t)acc_arr[z]; //ver o fator de escala aqui
						//aplicar função de ativação aqui
#else
//						printf("sending pixel number %d of filter %d \n", (i*fm_height)+j, z);
						tmpout = (quant_t)acc_arr[z];
						strm_out.write(tmpout);
#endif
						acc_arr[z] = 0;
					}

				}
			}
		}
	}
	return;
}

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, quant_t *weights>
#ifdef ARRAYS
void conv_layer_relu(quant_t in_feature_map[fm_height*fm_width*nbands], quant_t out_feature_map[fm_height*fm_width*nfilters]) {
#else
void conv_layer_relu(hls::stream<quant_t> &strm_in, hls::stream<quant_t> &strm_out) {
#endif

#ifndef ARRAYS
	quant_t tmpin, tmpout;
	quant_t pixel;
#endif
	int negatives = 0;

	//Convolution
	quant_mult acc = 0;
	quant_accum acc_arr[nfilters];
	int kernel_idx = 0;

	loop_inputx:
	for(count_t i = 0; i < fm_width; i++) {
		loop_inputy:
		for(count_t j = 0; j < fm_height; j++) {
			kernel_idx = 0;
			loop_bands:
			for(count_t k = 0; k < nbands; k++) {
#ifndef ARRAYS
				tmpin = strm_in.read();
				pixel = tmpin;
#endif
				loop_filters:
				for(count_t z = 0; z < nfilters; z++) {
#pragma HLS PIPELINE
					acc = 0;
#ifdef ARRAYS
					/* Input matrix index */
					count_t input_idx =
							k * (fm_width * fm_height) + (i * fm_height +  /* input row */
							j);                                      /* input column */
#endif
//							printf("%d IP - %d %d %f\n",z, k, (x*kernel_size) + y, (float)weights[kernel_idx]);
#ifdef ARRAYS
					acc += weights[kernel_idx] * in_feature_map[input_idx];
#else
					acc += weights[kernel_idx] *  pixel;
#endif
					kernel_idx++;

					if(k == 0)
						acc_arr[z] = acc;
					else
						acc_arr[z] += acc;
					if(k == nbands-1) {  //If it's the last band, put the accum in the output feature map and reset the accum array
#ifdef ARRAYS
						out_feature_map[z*fm_width*fm_height + i*fm_height + j] = (quant_t)acc_arr[z]; //ver o fator de escala aqui
						//aplicar função de ativação aqui
#else
//						printf("sending pixel number %d of filter %d \n", (i*fm_height)+j, z);

						tmpout = (quant_t)acc_arr[z];
						tmpout = tmpout > 0 ? tmpout : (quant_t)0;
						strm_out.write(tmpout);
#endif
						acc_arr[z] = 0;
					}

				}
			}
		}
	}
	return;
}
