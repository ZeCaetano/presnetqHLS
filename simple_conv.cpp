#include "simple_conv.h"
//#include "weights_k1.h"      //Weights for two k1 layers arranged by band
//#include "weights_k1_2.h"      //Weights for two k1 layers arranged by band for 48 filters in layer 2
//#include "weights_k1_3.h"      //Weights for two k1 layers arranged by band for a lot of bands and filters
//#include "weights_k1_4.h"      //Weights for two k1 layers arranged by band for a 80 and 96 filters
//#include "weights_k2.h"    //Weights for layer 2 k2 arranged by filter
//#include "weights_k2_2.h"  //Weights for layer 2 k2 arranged by bands
//#include "weights_k2_3.h"  //Weights for layer 2 k2 arranged by bands by every two pixels
//#include "weights_k2_4.h"  //Weights for layer 2 k2 arranged by bands by every two pixels for 48 filters on layer 2
//#include "weights_k2_5.h"  //Weights for layer 2 k2 arranged by bands by every two pixels for 96 filters on layer 1 and 2
#include "weights_k2_6.h"  //Weights for layer 2 k2 arranged by bands by every two pixels for 3 layers

void simple_conv(hls::stream<strmio_t> &strm_in, hls::stream<strmio_t> &strm_out) {

//#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=strm_in
#pragma HLS INTERFACE axis port=strm_out

	quant_reshp in_feature_map[INPUT1_MEM_SIZE/RESHP_FACTOR], out_feature_map[OUTPUT_MEM_SIZE/RESHP_FACTOR], m2_feature_map[OUT2_FM_MEM_SIZE/RESHP_FACTOR], m3_feature_map[OUT3_FM_MEM_SIZE/RESHP_FACTOR];//
	quant_t shortcut_fm[X1*Y1*Z1/RESHP_FACTOR], ds_ofm[XDS*YDS*ZDS/RESHP_FACTOR];
//	quant_t m1_feature_map[(X2)*(Y2)*Z2];
	quant_reshp m1_feature_map[2][((X2-1)*(Y2-1)*Z2/2)/RESHP_FACTOR];

#pragma HLS STREAM variable=in_feature_map type=PIPO depth=2
#pragma HLS STREAM variable=shortcut_fm type=PIPO depth=4
#pragma HLS STREAM variable=ds_ofm type=PIPO depth=4
#pragma HLS STREAM variable=m1_feature_map type=PIPO depth=2
#pragma HLS STREAM variable=m2_feature_map type=PIPO depth=2
#pragma HLS STREAM variable=m3_feature_map type=PIPO depth=2
#pragma HLS STREAM variable=out_feature_map type=PIPO depth=2

//#pragma HLS ARRAY_RESHAPE variable=in_feature_map type=cyclic factor=8
//#pragma HLS ARRAY_RESHAPE variable=m1_feature_map type=cyclic factor=8
//#pragma HLS ARRAY_RESHAPE variable=m2_feature_map type=cyclic factor=8
//#pragma HLS ARRAY_RESHAPE variable=m3_feature_map type=cyclic factor=8
//#pragma HLS ARRAY_RESHAPE variable=out_feature_map type=cyclic factor=8

//#pragma HLS ARRAY_PARTITION variable=in_feature_map type=cyclic factor=8
#pragma HLS ARRAY_RESHAPE variable=weights_l1 factor=4 type=cyclic
#pragma HLS ARRAY_RESHAPE variable=weights_l2 factor=4 type=cyclic
#pragma HLS ARRAY_RESHAPE variable=weights_l3 factor=8 type=cyclic

#pragma HLS DATAFLOW
	read_ifm(strm_in, in_feature_map, shortcut_fm);
//	average_pool<X1,Y1,XDS,YDS,Z1,KDS>(shortcut_fm, ds_ofm);
	conv_layer_k1_b4k2<0,X1,Y1,Z1,NF1, weights_l1, 4> (in_feature_map, m1_feature_map);
	conv_layer_k2<1,X2-1,Y2-1,Z2,NF2, X3, weights_l2, 4> (m1_feature_map, m2_feature_map);
	conv_layer_k1<2,X3,Y3,Z3,NF3, weights_l3, 8> (m2_feature_map, m3_feature_map);
//	conv_layer_k1<0,X1,Y1,Z1,NF1, weights_l1, 16> (in_feature_map, m1_feature_map);
//	conv_layer_k1<1,X2,Y2,Z2,NF2, weights_l2, 16> (m1_feature_map, m2_feature_map);
//	add_shortcut<X3,Y3,NF3,ZDS>(m3_feature_map, ds_ofm, out_feature_map);
	write_ofm(m3_feature_map, strm_out);

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
void read_ifm(hls::stream<strmio_t> &strm_in, quant_reshp in_feature_map[INPUT1_MEM_SIZE], quant_t shortcut_ifm[INPUT1_MEM_SIZE]){
#else
void read_ifm(hls::stream<strmio_t> &strm_in, hls::stream<quant_t> &in_feature_map){
#endif
	strmio_t tmpin;

#ifdef ARRAYS
	//read input fm
	for(int i = 0; i < INPUT1_MEM_SIZE/RESHP_FACTOR; i++) {
#pragma HLS PIPELINE II=4
		tmpin = strm_in.read();
		in_feature_map[i].range(3,0) = tmpin.data;
		shortcut_ifm[i*RESHP_FACTOR] = tmpin.data;
//		printf("%d   ", (int)(quant_t)(in_feature_map[i].range(3,0)));
		if(tmpin.last == 1) break;
		tmpin = strm_in.read();
		in_feature_map[i].range(7,4) = tmpin.data;
		shortcut_ifm[i*RESHP_FACTOR+1] = tmpin.data;
//		printf("%d: %d   ", (int)tmpin.data, (int)((quant_t)in_feature_map[i].range(7,4)));
		if(tmpin.last == 1) break;
		tmpin = strm_in.read();
		in_feature_map[i].range(11,8) = tmpin.data;
		shortcut_ifm[i*RESHP_FACTOR+2] = tmpin.data;
//		printf("%d: %d   ", (int)tmpin.data, (int)((quant_t)in_feature_map[i].range(11,8)));
		if(tmpin.last == 1) break;
		tmpin = strm_in.read();
		in_feature_map[i].range(15,12) = tmpin.data;
		shortcut_ifm[i*RESHP_FACTOR+3] = tmpin.data;
//		printf("%d: %d \n", (int)tmpin.data, (int)((quant_t)in_feature_map[i].range(15,12)));
		if(tmpin.last == 1) break;
//		tmpin = strm_in.read();
//		(quant_t)in_feature_map[i].range(19,16) = tmpin.data;
//		(quant_t)shortcut_ifm[i].range(19,16) = tmpin.data;
//		if(tmpin.last == 1) break;
//		tmpin = strm_in.read();
//		(quant_t)in_feature_map[i].range(23,20) = tmpin.data;
//		(quant_t)shortcut_ifm[i].range(23,20) = tmpin.data;
//		if(tmpin.last == 1) break;
//		tmpin = strm_in.read();
//		(quant_t)in_feature_map[i].range(27,24) = tmpin.data;
//		(quant_t)shortcut_ifm[i].range(27,24) = tmpin.data;
//		if(tmpin.last == 1) break;
//		tmpin = strm_in.read();
//		(quant_t)in_feature_map[i].range(31,28) = tmpin.data;
//		(quant_t)shortcut_ifm[i].range(31,28) = tmpin.data;

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
void write_ofm(quant_reshp ofm[OUTPUT_MEM_SIZE/RESHP_FACTOR], hls::stream<strmio_t> &strm_out) {
#else
void write_ofm(hls::stream<quant_t> &ofm, hls::stream<strmio_t> &strm_out, count_t n_pixels) {
#endif
	strmio_t tmpout;
#ifndef ARRAYS
	quant_t tmpin;
#endif
	//Write output fm to stream
	for(int i = 0; i < OUTPUT_MEM_SIZE/RESHP_FACTOR; i++){
#pragma HLS PIPELINE II=4
		tmpout.data = (quant_t)ofm[i].range(3,0);
//		printf("%d: %d   \n", (int)(quant_t)ofm[i].range(3,0), (int)tmpout.data);
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		tmpout.last = 0;
		strm_out.write(tmpout);
		tmpout.data = (quant_t)ofm[i].range(7,4);
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		tmpout.last = 0;
		strm_out.write(tmpout);
		tmpout.data = (quant_t)ofm[i].range(11,8);
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		tmpout.last = 0;
		strm_out.write(tmpout);
		if(i == (OUTPUT_MEM_SIZE/RESHP_FACTOR) - 1) tmpout.last = 1;
		else tmpout.last = 0;
		tmpout.data = (quant_t)ofm[i].range(15,12);
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		strm_out.write(tmpout);
//		tmpout.data = (quant_t)ofm[i].range(19,16);
//		tmpout.keep = 0xF;
//		tmpout.strb = 0xF;
//		strm_out.write(tmpout);
//		tmpout.data = (quant_t)ofm[i].range(23,20);
//		tmpout.keep = 0xF;
//		tmpout.strb = 0xF;
//		strm_out.write(tmpout);
//		tmpout.data = (quant_t)ofm[i].range(27,24);
//		tmpout.keep = 0xF;
//		tmpout.strb = 0xF;
//		strm_out.write(tmpout);
//		tmpout.data = (quant_t)ofm[i].range(31,28);
//		tmpout.keep = 0xF;
//		tmpout.strb = 0xF;
//		strm_out.write(tmpout);
	}
}


template<params_t fm_width, params_t fm_height, params_t output_width, params_t output_height, params_t nbands, params_t kernel_size>
void average_pool(quant_t in_feature_map[fm_width*fm_height*nbands], quant_t out_feature_map[output_height*output_width*nbands]){
	quant_accum accum = 0;
	quant_t avg = 0;
	ap_uint<3> div = kernel_size*kernel_size;

	for(int z = 0; z < nbands; z++){
		for (int i = 0; i < fm_width-1; i+=2) {
			for (int j = 0; j < fm_height-1; j+=2) {
				for(int k = 0; k < kernel_size; k++){
					for(int l = 0; l < kernel_size; l++){
#pragma HLS PIPELINE
//						accum += in_feature_map[(z*fm_width*fm_height) + (i+k)*fm_height+ j+l];
						accum += in_feature_map[(i+k)*nbands*fm_height + (j+l)*nbands + z];
						if(k == kernel_size-1 && l == kernel_size-1){
							avg = accum / div;
							out_feature_map[(i/2)*nbands*output_height+ (j/2)*nbands + z] = (quant_t) avg;
							accum = 0;
						}
					}
				}
			}
		}
	}
}

template<params_t fm_width, params_t fm_height, params_t nbands_conv, params_t nbands_shortcut>
void add_shortcut(quant_reshp conv_feature_map[fm_width*fm_height*nbands_conv], quant_t shortcut[fm_width*fm_height*nbands_shortcut], quant_reshp out_feature_map[fm_width*fm_height*nbands_conv]){
	int idx_conv = 0;
	int idx_shortcut = 0;
	quant_sum sum = 0;
	for (int i = 0; i < fm_width; i++) {
		for (int j = 0; j < fm_height; j++) {
			for(int z = 0; z < nbands_conv/RESHP_FACTOR; z++){
#pragma HLS PIPELINE
				if(z >= nbands_shortcut){
					sum = (quant_t)conv_feature_map[idx_conv].range(3,0);
					out_feature_map[idx_conv].range(3,0) = (quant_t) sum;
					sum = (quant_t)conv_feature_map[idx_conv].range(7,4);
					out_feature_map[idx_conv].range(7,4) = (quant_t) sum;
					sum = (quant_t)conv_feature_map[idx_conv].range(11,8);
					out_feature_map[idx_conv].range(11,8) = (quant_t) sum;
					sum = (quant_t)conv_feature_map[idx_conv].range(15,12);
					out_feature_map[idx_conv].range(15,12) = (quant_t) sum;
					idx_conv++;
				}
				else {
					sum = (quant_t)conv_feature_map[idx_conv].range(3,0) + shortcut[idx_shortcut];
					out_feature_map[idx_conv].range(3,0) = (quant_t) sum;
					idx_shortcut++;
					sum = (quant_t)conv_feature_map[idx_conv].range(7,4) + shortcut[idx_shortcut];
					out_feature_map[idx_conv].range(7,4) = (quant_t) sum;
					idx_shortcut++;
					sum = (quant_t)conv_feature_map[idx_conv].range(11,8) + shortcut[idx_shortcut];
					out_feature_map[idx_conv].range(11,8) = (quant_t) sum;
					idx_shortcut++;
					sum = (quant_t)conv_feature_map[idx_conv].range(15,12) + shortcut[idx_shortcut];
					out_feature_map[idx_conv].range(15,12) = (quant_t) sum;
					idx_conv++;
					idx_shortcut++;
				}
			}
		}
	}
}

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, quant_t *weights, params_t PE>
#ifdef ARRAYS
void conv_layer_k1(quant_reshp in_feature_map[fm_height*fm_width*nbands], quant_reshp out_feature_map[fm_height*fm_width*nfilters]) {
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
	ap_uint<3> range_idx = 0;

	loop_inputx:
	for(int i = 0; i < fm_width; i++) {
		loop_inputy:
		for(int j = 0; j < fm_height; j++) {
			kernel_idx = 0;
			loop_filters:
			for(int k = 0; k < nfilters; k++) {
//				for(int z = 0; z < nbands; z+=PE) {
//				#pragma HLS PIPELINE
//					for(int p = 0; p < PE/RESHP_FACTOR; p++) {
				loop_bands:
				for(int z = 0; z < nbands/RESHP_FACTOR; z+=PE) {
#pragma HLS PIPELINE II=4
					for(int p = 0; p < PE; p++) {
						acc += (quant_t)weights[kernel_idx] * (quant_t)in_feature_map[input_idx].range(3,0);
						kernel_idx++;
						acc += (quant_t)weights[kernel_idx] * (quant_t)in_feature_map[input_idx].range(7,4);
						kernel_idx++;
						acc += (quant_t)weights[kernel_idx] * (quant_t)in_feature_map[input_idx].range(11,8);
						kernel_idx++;
						acc += (quant_t)weights[kernel_idx] * (quant_t)in_feature_map[input_idx].range(15,12);
						kernel_idx++;
						input_idx++;
//						printf("%d + %d*4 = %d\n", z,  p*RESHP_FACTOR,z + (p*RESHP_FACTOR));
//						if(z + (p*RESHP_FACTOR) == nbands-RESHP_FACTOR) {
						if(z + p == nbands/RESHP_FACTOR-1) {
	//						count_t output_idx = k*(fm_width)*(fm_height) + i*(fm_height) + j;
							if(range_idx == 0) {
								out_feature_map[output_idx].range(3,0) = (quant_t)acc; //ver o fator de escala aqui
//								printf("idx: %d; acc: %d; output: %d\n", output_idx, (int)(quant_t)acc, (int)(quant_t)out_feature_map[output_idx].range(3,0));
								range_idx++;
							}
							else if(range_idx == 1) {
								out_feature_map[output_idx].range(7,4) = (quant_t)acc; //ver o fator de escala aqui
								range_idx++;
							}
							else if(range_idx == 2) {
								out_feature_map[output_idx].range(11,8) = (quant_t)acc; //ver o fator de escala aqui
								range_idx++;
							}
							else if(range_idx == 3) {
								out_feature_map[output_idx].range(15,12) = (quant_t)acc; //ver o fator de escala aqui
								range_idx = 0;
								output_idx++;
							}
							acc = 0;
							if(k != nfilters-1) input_idx -= nbands/RESHP_FACTOR;
							break;
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
void conv_layer_k1_b4k2(quant_reshp in_feature_map[fm_height*fm_width*nbands], quant_reshp out_feature_map[2][(fm_height-1)*(fm_width-1)*nfilters/2/RESHP_FACTOR]) {
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
	ap_uint<3> range_idx = 0;

	loop_inputx:
	for(int i = 0; i < fm_width-1; i++) {
		loop_inputy:
		for(int j = 0; j < fm_height; j++) {
			kernel_idx = 0;
			loop_filters:
			for(int k = 0; k < nfilters; k++) {
				loop_bands:
				for(int z = 0; z < nbands/RESHP_FACTOR; z+=PE) {
#pragma HLS PIPELINE II=4
					for(int p = 0; p < PE; p++) {
						if(j == fm_height-1){
//							printf("J MAIOR\n");
							if(k == 0 && z == 0 && p == 0)
								input_idx += nbands/RESHP_FACTOR;
	//						z = nbands;
	//						k = nfilters;
						}
						else {
							acc += weights[kernel_idx] * (quant_t)in_feature_map[input_idx].range(3,0);
//							printf("idx: %d\nacc = %d * %d\n", input_idx,(int)weights[kernel_idx], (int)(quant_t)in_feature_map[input_idx].range(3,0));
							kernel_idx++;
							acc += weights[kernel_idx] * (quant_t)in_feature_map[input_idx].range(7,4);
//							printf("acc = %d * %d\n", (int)weights[kernel_idx], (int)(quant_t)in_feature_map[input_idx].range(7,4));
							kernel_idx++;
							acc += weights[kernel_idx] * (quant_t)in_feature_map[input_idx].range(11,8);
//							printf("acc = %d * %d\n", (int)weights[kernel_idx], (int)(quant_t)in_feature_map[input_idx].range(11,8));
							kernel_idx++;
							acc += weights[kernel_idx] * (quant_t)in_feature_map[input_idx].range(15,12);
//							printf("acc = %d * %d\n", (int)weights[kernel_idx], (int)(quant_t)in_feature_map[input_idx].range(15,12));
							kernel_idx++;
							input_idx++;
							if(z + p == nbands/RESHP_FACTOR-1) {
								if(i % 2 == 0) {
									if(range_idx == 0) {
										out_feature_map[0][output_idx_even].range(3,0) = (quant_t)acc; //ver o fator de escala aqui
//										printf("idx: %d; acc: %d; output: %d\n", output_idx_even, (int)(quant_t)acc, (int)(quant_t)out_feature_map[0][output_idx_even].range(3,0));
										range_idx++;
									}
									else if(range_idx == 1) {
										out_feature_map[0][output_idx_even].range(7,4) = (quant_t)acc; //ver o fator de escala aqui
										range_idx++;
									}
									else if(range_idx == 2) {
										out_feature_map[0][output_idx_even].range(11,8) = (quant_t)acc; //ver o fator de escala aqui
										range_idx++;
									}
									else if(range_idx == 3) {
										out_feature_map[0][output_idx_even].range(15,12) = (quant_t)acc; //ver o fator de escala aqui
										range_idx = 0;
										output_idx_even++;
									}
	//								printf("index: [0][%d] - %d\n", output_idx_even, (int)out_feature_map[0][output_idx_even]);
	//								printf("index: [0][%d] - %d\n", output_idx_even, (int)acc);
								}
								else {
									if(range_idx == 0) {
										out_feature_map[1][output_idx_odd].range(3,0) = (quant_t)acc; //ver o fator de escala aqui
										range_idx++;
									}
									else if(range_idx == 1) {
										out_feature_map[1][output_idx_odd].range(7,4) = (quant_t)acc; //ver o fator de escala aqui
										range_idx++;
									}
									else if(range_idx == 2) {
										out_feature_map[1][output_idx_odd].range(11,8) = (quant_t)acc; //ver o fator de escala aqui
										range_idx++;
									}
									else if(range_idx == 3) {
										out_feature_map[1][output_idx_odd].range(15,12) = (quant_t)acc; //ver o fator de escala aqui
										range_idx = 0;
										output_idx_odd++;
									}
	//								printf("index: [1][%d] - %d\n", output_idx_odd, (int)out_feature_map[1][output_idx_odd]);
								}

								if(k != nfilters-1) input_idx -= nbands/RESHP_FACTOR;
								acc = 0;
								break;
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
void conv_layer_k2(quant_reshp in_feature_map[2][fm_height*fm_width*nbands/2/RESHP_FACTOR], quant_reshp out_feature_map[(fm_height/2)*(fm_width/2)*nfilters]) {
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
	ap_uint<3> range_idx = 0;

//#pragma HLS ARRAY_RESHAPE variable=in_feature_map type=cyclic factor=2

	loop_inputx:
	for(int i = 0; i < fm_width-1; i+=2) {
		loop_inputy:
		for(int j = 0; j < fm_height-1; j+=2) {
			kernel_idx = 0;
			loop_filters:
			for(int k = 0; k < nfilters; k++) {
				loop_bands:
				for(int z = 0; z < nbands*2/RESHP_FACTOR; z+=PE) {
#pragma HLS PIPELINE II=4
					for(int p = 0; p < PE; p++) {
						acc_even += weights[kernel_idx] * (quant_t)in_feature_map[0][input_idx].range(3,0);
						kernel_idx++;
						acc_odd += weights[kernel_idx] * (quant_t)in_feature_map[1][input_idx].range(3,0);
						kernel_idx++;
						acc_even += weights[kernel_idx] * (quant_t)in_feature_map[0][input_idx].range(7,4);
						kernel_idx++;
						acc_odd += weights[kernel_idx] * (quant_t)in_feature_map[1][input_idx].range(7,4);
						kernel_idx++;
						acc_even += weights[kernel_idx] * (quant_t)in_feature_map[0][input_idx].range(11,8);
						kernel_idx++;
						acc_odd += weights[kernel_idx] * (quant_t)in_feature_map[1][input_idx].range(11,8);
						kernel_idx++;
						acc_even += weights[kernel_idx] * (quant_t)in_feature_map[0][input_idx].range(15,12);
						kernel_idx++;
						acc_odd += weights[kernel_idx] * (quant_t)in_feature_map[1][input_idx].range(15,12);
						kernel_idx++;
						input_idx++;

						if(z + p == (nbands*2)/RESHP_FACTOR-1) {
							acc_even += acc_odd;
							if(range_idx == 0) {
								out_feature_map[output_idx].range(3,0) = (quant_t)acc_even; //ver o fator de escala aqui
								acc_even = 0;
								acc_odd = 0;
								range_idx++;
							}
							else if(range_idx == 1) {
								out_feature_map[output_idx].range(7,4) = (quant_t)acc_even; //ver o fator de escala aqui
								acc_even = 0;
								acc_odd = 0;
								range_idx++;
							}
							else if(range_idx == 2) {
								out_feature_map[output_idx].range(11,8) = (quant_t)acc_even; //ver o fator de escala aqui
								acc_even = 0;
								acc_odd = 0;
								range_idx++;
							}
							else if(range_idx == 3) {
								out_feature_map[output_idx].range(15,12) = (quant_t)acc_even; //ver o fator de escala aqui
								acc_even = 0;
								acc_odd = 0;
								range_idx = 0;
								output_idx++;
							}
							//aplicar função de ativação aqui
							if(k != nfilters-1) input_idx -= nbands*2/RESHP_FACTOR;
							break;
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
