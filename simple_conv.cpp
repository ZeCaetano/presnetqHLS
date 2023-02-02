#include "simple_conv.h"
//#include "weights_k1.h"      //Weights for two k1 layers arranged by band
#include "weights_k1_2.h"      //Weights for two k1 layers arranged by band for 48 filters in layer 2
//#include "weights_k1_3.h"      //Weights for two k1 layers arranged by band for a lot of bands and filters
//#include "weights_k1_4.h"      //Weights for two k1 layers arranged by band for a 80 and 96 filters
//#include "weights_k2.h"    //Weights for layer 2 k2 arranged by filter
//#include "weights_k2_2.h"  //Weights for layer 2 k2 arranged by bands
//#include "weights_k2_3.h"  //Weights for layer 2 k2 arranged by bands by every two pixels
//#include "weights_k2_4.h"  //Weights for layer 2 k2 arranged by bands by every two pixels for 48 filters on layer 2
//#include "weights_k2_5.h"  //Weights for layer 2 k2 arranged by bands by every two pixels for 96 filters on layer 1 and 2

void simple_conv(hls::stream<strmio_t> &strm_in, hls::stream<strmio_t> &strm_out) {

//#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=strm_in
#pragma HLS INTERFACE axis port=strm_out

	quant_t in_feature_map[X1*Y1*Z1], out_feature_map[X3*Y3*NF2]; //, m2_feature_map[X3*Y3*NF2];//, ds_feature_map[XDS*YDS*Z1];
	quant_t m1_feature_map[(X2)*(Y2)*Z2];
//	quant_t m1_feature_map[2][(X2-1)*(Y2-1)*Z2/2];

#pragma HLS STREAM variable=in_feature_map type=PIPO depth=2
#pragma HLS STREAM variable=m1_feature_map type=PIPO depth=2
#pragma HLS STREAM variable=out_feature_map type=PIPO depth=2

//#pragma HLS ARRAY_RESHAPE variable=in_feature_map type=cyclic factor=8
//#pragma HLS ARRAY_RESHAPE variable=m1_feature_map type=cyclic factor=8
//#pragma HLS ARRAY_RESHAPE variable=out_feature_map type=cyclic factor=8

#pragma HLS ARRAY_RESHAPE variable=weights_l1 factor=8 type=cyclic
#pragma HLS ARRAY_RESHAPE variable=weights_l2 factor=8 type=cyclic

#pragma HLS DATAFLOW
	read_ifm(strm_in, in_feature_map);
//	conv_layer_k1_b4k2<0,X1,Y1,Z1,NF1, weights_l1, 16> (in_feature_map, m1_feature_map);
//	conv_layer_k2<1,X2-1,Y2-1,Z2,NF2, X3, weights_l2, 8> (m1_feature_map, out_feature_map);
	conv_layer_k1<0,X1,Y1,Z1,NF1, weights_l1, 8> (in_feature_map, m1_feature_map);
	conv_layer_k1<1,X2,Y2,Z2,NF2, weights_l2, 8> (m1_feature_map, out_feature_map);
//	conv_layer_k1<1,X2,Y2,Z2,NF2, weights_l2, 4> (m2_feature_map, out_feature_map);
	write_ofm(out_feature_map, strm_out);

}


//void dataflow_func(hls::stream<strmio_t> &strm_in, hls::stream<strmio_t> &strm_out){
//
//#ifdef ARRAYS
//	quant_t in_feature_map[X1*Y1*Z1], out_feature_map[X3*Y3*NF2]; //, m2_feature_map[X3*Y3*NF2], ds_feature_map[XDS*YDS*Z1];
//	quant_t m1_feature_map[(X2)*(Y2)*Z2];
////	quant_t m1_feature_map[2][(X2-1)*(Y2-1)*Z2/2];
//
//#pragma HLS STREAM variable=in_feature_map type=PIPO depth=2
//#pragma HLS STREAM variable=m1_feature_map type=PIPO depth=2
//#pragma HLS STREAM variable=out_feature_map type=PIPO depth=2
//
//#pragma HLS DATAFLOW
//	read_ifm(strm_in, in_feature_map);
//	conv_layer_k1<0,X1,Y1,Z1,NF1, weights_l1, 1> (in_feature_map, m1_feature_map);
//	conv_layer_k1<1,X2,Y2,Z2,NF2, weights_l2, 1> (m1_feature_map, out_feature_map);
//	write_ofm(out_feature_map, strm_out);
//
//	//	average_pool<X1,Y1,XDS,YDS,Z1,KDS>(in_feature_map, ds_feature_map, in_cpy);
////	add_shortcut<X3,Y3,Z3,Z1>(m2_feature_map, ds_feature_map, out_feature_map);
//
//#else
//	hls::stream<quant_t> m0, m1, m2;
//
//	#pragma HLS STREAM variable=m0 depth=2
//	#pragma HLS STREAM variable=m1 depth=2
//	#pragma HLS STREAM variable=m2 depth=2
//
//	read_ifm(strm_in, m0);
//
//	conv_layer_k1<0,X1,Y1,Z1,NF1, weights_l1> (m0, m1);
//	conv_layer_k1<1,X2,Y2,Z2,NF2, weights_l2> (m1, m2);
//
//	write_ofm(m2, strm_out, X2*Y2*NF2);
//#endif
//}


#ifdef ARRAYS
void read_ifm(hls::stream<strmio_t> &strm_in, quant_t in_feature_map[INPUT1_MEM_SIZE]){
#else
void read_ifm(hls::stream<strmio_t> &strm_in, hls::stream<quant_t> &in_feature_map){
#endif
	strmio_t tmpin;

#ifdef ARRAYS
	//read input fm
	for(int i = 0; i < INPUT1_MEM_SIZE; i++) {
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

//void read_stream(hls::stream<strmio_t> &strm_in, quant_t *weights_l1, quant_t *weights_l2) {
//
//	strmio_t tmpin;
//	quant_t tmpout;
//
//	//Read weights for layer 1
//	for(int i = 0; i < LAYER1_WEIGHTS; i++) {
//		tmpin = strm_in.read();
////		printf("Receive %f\n", (float)tmpin.data);
//		weights_l1[i] = tmpin.data;
////		if(layer_id == 1) printf("%d  %f-%d\n",i, weights[i], tmpin.last);
//		if(tmpin.last == 1) break;
//	}
//
//	//Read weights for layer 2
//	for(int i = 0; i < LAYER2_WEIGHTS; i++) {
//		tmpin = strm_in.read();
////		printf("Receive %f\n", (float)tmpin.data);
//		weights_l2[i] = tmpin.data;
////		if(layer_id == 1) printf("%d  %f-%d\n",i, weights[i], tmpin.last);
//		if(tmpin.last == 1) break;
//	}
//}

#ifdef ARRAYS
void write_ofm(quant_t ofm[OUTPUT_MEM_SIZE], hls::stream<strmio_t> &strm_out) {
#else
void write_ofm(hls::stream<quant_t> &ofm, hls::stream<strmio_t> &strm_out, count_t n_pixels) {
#endif
	strmio_t tmpout;
#ifndef ARRAYS
	quant_t tmpin;
#endif
//#pragma HLS ARRAY_RESHAPE variable=ofm type=cyclic factor=1
	//Write output fm to stream
	for(int i = 0; i < OUTPUT_MEM_SIZE; i++){
//#pragma HLS UNROLL factor=4
		if(i == OUTPUT_MEM_SIZE - 1) tmpout.last = 1;
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


template<params_t fm_width, params_t fm_height, params_t output_width, params_t output_height, params_t nbands, params_t kernel_size>
void average_pool(quant_t in_feature_map[fm_width*fm_height*nbands], quant_t out_feature_map[output_height*output_width*nbands]){
	quant_sum accum = 0;
	quant_t avg = 0;
	ap_uint<3> div = kernel_size*kernel_size;

	for(int z = 0; z < nbands; z++){
		for (int i = 0; i < fm_width-1; i+=2) {
			for (int j = 0; j < fm_height-1; j+=2) {
				accum = 0;
				for(int k = 0; k < kernel_size; k++){
					for(int l = 0; l < kernel_size; l++){
#pragma HLS PIPELINE
						accum += in_feature_map[(z*fm_width*fm_height) + (i+k)*fm_height+ j+l];
					}
				}
				avg = accum / div;
				out_feature_map[z*output_width*output_height+ (i/2)*output_height+ (j/2)] = (quant_t) avg;
			}
		}
	}
}

template<params_t fm_width, params_t fm_height, params_t nbands_conv, params_t nbands_shortcut>
void add_shortcut(quant_t conv_feature_map[fm_width*fm_height*nbands_conv], quant_t shortcut[fm_width*fm_height*nbands_shortcut], quant_t out_feature_map[fm_width*fm_height*nbands_conv]){
	for(int z = 0; z < nbands_conv; z++){
		for (int i = 0; i < fm_width; i++) {
			for (int j = 0; j < fm_height; j++) {
#pragma HLS PIPELINE
				if(z > nbands_shortcut)
					out_feature_map[z*fm_width*fm_height+ i*fm_height+ j] = conv_feature_map[z*fm_width*fm_height+ i*fm_height+ j];
				else
					out_feature_map[z*fm_width*fm_height+ i*fm_height+ j] = (quant_t) (conv_feature_map[z*fm_width*fm_height+ i*fm_height+ j] + shortcut[z*fm_width*fm_height+ i*fm_height+ j]);

			}
		}
	}
}

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, quant_t *weights, params_t PE>
#ifdef ARRAYS
void conv_layer_k1(quant_t in_feature_map[fm_height*fm_width*nbands], quant_t out_feature_map[fm_height*fm_width*nfilters]) {
#else
void conv_layer_k1_b4k2(hls::stream<quant_t> &strm_in, hls::stream<quant_t> &strm_out) {
#endif

#ifndef ARRAYS
	quant_t tmpin, tmpout;
	quant_t pixel;
#endif
//#pragma HLS ARRAY_RESHAPE variable=in_feature_map type=cyclic factor=4
//#pragma HLS ARRAY_RESHAPE variable=out_feature_map type=cyclic factor=4
	//Convolution
	quant_accum acc = 0;
	int kernel_idx = 0;
	int input_idx = 0;
	int output_idx = 0;

	loop_inputx:
	for(int i = 0; i < fm_width; i++) {
		loop_inputy:
		for(int j = 0; j < fm_height; j++) {
			kernel_idx = 0;
			loop_filters:
			for(int k = 0; k < nfilters; k++) {
				loop_bands:
				for(int z = 0; z < nbands; z+=PE) {
#pragma HLS PIPELINE
					for(int p = 0; p < PE; p++) {
						acc += weights[kernel_idx] * in_feature_map[input_idx];
						kernel_idx++;
						input_idx++;
						if(z == nbands-PE && p == PE-1) {
	//						count_t output_idx = k*(fm_width)*(fm_height) + i*(fm_height) + j;
							out_feature_map[output_idx] = (quant_t)acc; //ver o fator de escala aqui
							output_idx++;
							//aplicar função de ativação aqui
							acc = 0;
							if(k != nfilters-1) input_idx -= nbands;
						}
					}
				}
			}
		}
	}
	return;
}
//Convolutional layer to be aplied before a convolution with kernel and stride 2, that will clear the last row and column from the outputs
template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, quant_t *weights, params_t PE>
#ifdef ARRAYS
void conv_layer_k1_b4k2(quant_t in_feature_map[fm_height*fm_width*nbands], quant_t out_feature_map[2][(fm_height-1)*(fm_width-1)*nfilters/2]) {
#else
void conv_layer_k1_b4k2(hls::stream<quant_t> &strm_in, hls::stream<quant_t> &strm_out) {
#endif

#ifndef ARRAYS
	quant_t tmpin, tmpout;
	quant_t pixel;
#endif

	//Convolution
	quant_accum acc = 0;
	int kernel_idx = 0;
	int input_idx = 0;
	int output_idx_even = 0;
	int output_idx_odd = 0;


	loop_inputx:
	for(int i = 0; i < fm_width-1; i++) {
		loop_inputy:
		for(int j = 0; j < fm_height; j++) {
			kernel_idx = 0;
			loop_filters:
			for(int k = 0; k < nfilters; k++) {
				loop_bands:
				for(int z = 0; z < nbands; z+=PE) {
#pragma HLS PIPELINE
					for(int p = 0; p < PE; p++) {
						if(j == fm_height-1){
							if(k == 0 && z == 0 && p == 0)
								input_idx += nbands;
	//						z = nbands;
	//						k = nfilters;
						}
						else {
							quant_accum temp = weights[kernel_idx] * in_feature_map[input_idx];
							acc += temp;
	//#pragma HLS BIND_OP variable=temp op=mul
							kernel_idx++;
							input_idx++;
							if(z == nbands-PE && p == PE-1) {
	//	//						count_t output_idx = k*(fm_width)*(fm_height) + i*(fm_height) + j;
	//							out_feature_map[output_idx] = (quant_t)acc; //ver o fator de escala aqui
	//							output_idx++;
	//							//aplicar função de ativação aqui

								if(((input_idx-1)/nbands/fm_width) % 2 == 0){
									out_feature_map[0][output_idx_even] = (quant_t)acc; //ver o fator de escala aqui
	//								printf("index: [0][%d] - %d\n", output_idx_even, (int)out_feature_map[0][output_idx_even]);
	//								printf("index: [0][%d] - %d\n", output_idx_even, (int)acc);
									output_idx_even++;
								}
								else {
									out_feature_map[1][output_idx_odd] = (quant_t)acc; //ver o fator de escala aqui
	//								printf("index: [1][%d] - %d\n", output_idx_odd, (int)out_feature_map[1][output_idx_odd]);
									output_idx_odd++;
								}

								if(k != nfilters-1) input_idx -= nbands;
								acc = 0;
							}
						}
					}
				}
			}
		}
	}
	return;
}

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t output_dim,quant_t *weights, params_t PE>
#ifdef ARRAYS
void conv_layer_k2(quant_t in_feature_map[2][fm_height*fm_width*nbands/2], quant_t out_feature_map[(fm_height/2)*(fm_width/2)*nfilters]) {
#else
void conv_layer_k2(hls::stream<quant_t> &strm_in, hls::stream<quant_t> &strm_out) {
#endif

#ifndef ARRAYS
	quant_t tmpin, tmpout;
	quant_t pixel;
#endif
	//Convolution with stride 2
	quant_accum acc_even = 0, acc_odd = 0;
	ap_uint<2> kernel_size = 2;
	int kernel_idx = 0;
	int input_idx = 0;
	int output_idx = 0;

//#pragma HLS ARRAY_RESHAPE variable=in_feature_map type=cyclic factor=2

	loop_inputx:
	for(int i = 0; i < fm_width-1; i+=2) {
		loop_inputy:
		for(int j = 0; j < fm_height-1; j+=2) {
			kernel_idx = 0;
			loop_filters:
			for(int k = 0; k < nfilters; k++) {
				loop_bands:
				for(int z = 0; z < nbands*2; z+=PE) {
#pragma HLS PIPELINE
					for(int p = 0; p < PE; p++) {
						acc_even += weights[kernel_idx] * in_feature_map[0][input_idx];
						kernel_idx++;
						acc_odd += weights[kernel_idx] * in_feature_map[1][input_idx];
						kernel_idx++;
						input_idx++;

						if(z == (nbands*2)-PE && p == PE-1) {
							acc_even += acc_odd;
	//						out_feature_map[k*output_dim*output_dim + (i/2)*output_dim+ (j/2)] = (quant_t)acc_even; //ver o fator de escala aqui
							out_feature_map[output_idx] = (quant_t)acc_even; //ver o fator de escala aqui
							output_idx++;
							acc_even = 0;
							acc_odd = 0;
							//aplicar função de ativação aqui
							if(k != nfilters-1) input_idx -= nbands*2;
						}
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
	for(int i = 0; i < fm_width; i++) {
		loop_inputy:
		for(int j = 0; j < fm_height; j++) {
			kernel_idx = 0;
			loop_bands:
			for(int z = 0; z < nbands; z++) {
#ifndef ARRAYS
				tmpin = strm_in.read();
				pixel = tmpin;
#endif
				loop_filters:
				for(int k = 0; k < nfilters; k++) {
#pragma HLS PIPELINE
					acc = 0;
#ifdef ARRAYS
					/* Input matrix index */
					count_t input_idx =
							z * (fm_width * fm_height) + (i * fm_height +  /* input row */
							j);                                      /* input column */
#endif
//							printf("%d IP - %d %d %f\n",z, k, (x*kernel_size) + y, (float)weights[kernel_idx]);
#ifdef ARRAYS
					acc += weights[kernel_idx] * in_feature_map[input_idx];
#else
					acc += weights[kernel_idx] *  pixel;
#endif
					kernel_idx++;

					if(z == 0)
						acc_arr[k] = acc;
					else
						acc_arr[k] += acc;
					if(z == nbands-1) {  //If it's the last band, put the accum in the output feature map and reset the accum array
#ifdef ARRAYS
						out_feature_map[z*fm_width*fm_height + i*fm_height + j] = (quant_t)acc_arr[k]; //ver o fator de escala aqui
						//aplicar função de ativação aqui
#else
//						printf("sending pixel number %d of filter %d \n", (i*fm_height)+j, z);

						tmpout = (quant_t)acc_arr[k];
						tmpout = tmpout > 0 ? tmpout : (quant_t)0;
						strm_out.write(tmpout);
#endif
						acc_arr[k] = 0;
					}

				}
			}
		}
	}
	return;
}
