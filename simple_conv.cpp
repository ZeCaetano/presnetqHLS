#include "simple_conv.h"
//#include "weights_k1.h"        //Weights for two k1 layers arranged by band
//#include "weights_k1_2.h"      //Weights for two k1 layers arranged by band for 48 filters in layer 2
//#include "weights_k1_3.h"      //Weights for two k1 layers arranged by band for a lot of bands and filters
//#include "weights_k1_4.h"      //Weights for two k1 layers arranged by band for a 80 and 96 filters
//#include "weights_k2.h"        //Weights for layer 2 k2 arranged by filter
//#include "weights_k2_2.h"      //Weights for layer 2 k2 arranged by bands
//#include "weights_k2_3.h"      //Weights for layer 2 k2 arranged by bands by every two pixels
//#include "weights_k2_4.h"      //Weights for layer 2 k2 arranged by bands by every two pixels for 48 filters on layer 2
//#include "weights_k2_5.h"      //Weights for layer 2 k2 arranged by bands by every two pixels for 96 filters on layer 1 and 2
//#include "weights_k2_6.h"        //Weights for layer 2 k2 arranged by bands by every two pixels for 3 layers
//#include "weights_reshaped.h"    //Weights reshaped with a factor of 4
//#include "weights_reshaped_2.h"    //Weights reshaped with a factor of 4 with 64 bands on the last layer
//#include "weights_reshaped_3.h"    //Weights reshaped with a factor of 8 with 64 bands and 168 filters on the last layer
//#include "weights_reshaped_4.h"    //Weights reshaped with a factor of 4 with 64 bands and 168 filters on the last layer
//#include "weights_reshaped_5.h"    //Weights reshaped for 8x2 quantization with a factor of 8 with 64 bands and 168 filters on the last layer
#include "weights_reshaped_7.h"    //Weights reshaped for 8x2 quantization with a factor of 8 with 64 bands and 168 filters on the last layer
//#include "weights_reshaped_8.h"    //Weights reshaped with fully connected layer for 8x2 quantization with a factor of 8 with 64 bands and 168 filters on the last layer


void simple_conv(hls::stream<strmio_t> &strm_in, hls::stream<strmio_t> &strm_out) {

#pragma HLS INTERFACE axis port=strm_in
#pragma HLS INTERFACE axis port=strm_out

	act_reshp in_feature_map[INPUT1_MEM_SIZE/RESHP_FACTOR], m4_feature_map[OUTPUT_MEM_SIZE/RESHP_FACTOR], m2_feature_map[OUT2_FM_MEM_SIZE/RESHP_FACTOR], m3_feature_map[OUT3_FM_MEM_SIZE/RESHP_FACTOR];//
	act_reshp m5_feature_map[OUTDS2_FM_MEM_SIZE/RESHP_FACTOR], out_feature_map[NCLASSES/RESHP_FACTOR];
	act_reshp shortcut_fm[X1*Y1*Z1/RESHP_FACTOR], ds_ofm[XDS*YDS*ZDS/RESHP_FACTOR];
//	quant_act m1_feature_map[(X2)*(Y2)*Z2];
//	act_reshp m1_feature_map[2][((X2-1)*(Y2-1)*Z2/2)/RESHP_FACTOR];
	act_reshp m1_feature_map[2][((X2)*(Y2)*Z2/2)/RESHP_FACTOR];

#pragma HLS BIND_STORAGE variable=m1_feature_map type=RAM_1P
#pragma HLS BIND_STORAGE variable=in_feature_map type=RAM_1P
#pragma HLS BIND_STORAGE variable=m2_feature_map type=RAM_1P

#pragma HLS STREAM variable=in_feature_map type=PIPO depth=2
#pragma HLS STREAM variable=shortcut_fm type=PIPO depth=2
#pragma HLS STREAM variable=ds_ofm type=PIPO depth=4
#pragma HLS STREAM variable=m1_feature_map type=PIPO depth=2
#pragma HLS STREAM variable=m2_feature_map type=PIPO depth=2
#pragma HLS STREAM variable=m3_feature_map type=PIPO depth=2
#pragma HLS STREAM variable=out_feature_map type=PIPO depth=2

#pragma HLS DATAFLOW
	read_ifm(strm_in, in_feature_map, shortcut_fm);
	average_pool<X1,Y1,XDS,YDS,Z1,KDS>(shortcut_fm, ds_ofm);
//	write_dsofm(ds_ofm, strm_out);
	conv_layer_k1_b4k2_x4<0,X1,Y1,Z1,NF1, weights_l1, 8, false> (in_feature_map, m1_feature_map);
	conv_layer_k2<1,X2,Y2,Z2,NF2, X3, weights_l2, 8, false> (m1_feature_map, m2_feature_map);
	conv_layer_k1<2,X3,Y3,Z3,NF3, weights_l3,8, false> (m2_feature_map, m3_feature_map);
	add_shortcut<X3,Y3,NF3,ZDS>(m3_feature_map, ds_ofm, m4_feature_map);
//	average_pool<X3,Y3,XDS2,YDS2,NF3,KDS>(m4_feature_map, m5_feature_map);
//	fully_connected<NF3,NCLASSES, weights_fc, bias_fc>(m5_feature_map, out_feature_map);
	write_ofm(m4_feature_map, strm_out);
}


void read_ifm(hls::stream<strmio_t> &strm_in, act_reshp in_feature_map[INPUT1_MEM_SIZE/RESHP_FACTOR], act_reshp shortcut_ifm[INPUT1_MEM_SIZE/RESHP_FACTOR]){
	strmio_t tmpin;
	act_reshp tmp = 0;

	//read input fm
	for(int i = 0; i < INPUT1_MEM_SIZE/RESHP_FACTOR; i++) {
#pragma HLS PIPELINE II=8
		tmpin = strm_in.read();
		tmp.range(7,0) = tmpin.data;
//		printf("%d   ", (int)(quant_act)(in_feature_map[i].range(7,0)));
		if(tmpin.last == 1) break;
		tmpin = strm_in.read();
		tmp.range(15,8) = tmpin.data;
//		printf("%d: %d   ", (int)tmpin.data, (int)((quant_act)in_feature_map[i].range(15,8)));
		if(tmpin.last == 1) break;
		tmpin = strm_in.read();
		tmp.range(23,16) = tmpin.data;
//		printf("%d: %d   ", (int)tmpin.data, (int)((quant_act)in_feature_map[i].range(23,16)));
		if(tmpin.last == 1) break;
		tmpin = strm_in.read();
		tmp.range(31,24) = tmpin.data;
		if(tmpin.last == 1) break;
		tmpin = strm_in.read();
		tmp.range(39,32) = tmpin.data;
//		printf("%d: %d   ", (int)tmpin.data, (int)((quant_act)in_feature_map[i].range(23,16)));
		if(tmpin.last == 1) break;
		tmpin = strm_in.read();
		tmp.range(47,40) = tmpin.data;
		//printf("%d: %d   ", (int)tmpin.data, (int)((quant_act)in_feature_map[i].range(23,16)));
		if(tmpin.last == 1) break;
		tmpin = strm_in.read();
		tmp.range(55,48) = tmpin.data;
		//printf("%d: %d   ", (int)tmpin.data, (int)((quant_act)in_feature_map[i].range(23,16)));
		if(tmpin.last == 1) break;
		tmpin = strm_in.read();
		tmp.range(63,56) = tmpin.data;
		in_feature_map[i] = tmp;
		shortcut_ifm[i] = tmp;
		//printf("%d: %d   ", (int)tmpin.data, (int)((quant_act)in_feature_map[i].range(23,16)));
		if(tmpin.last == 1) break;

//		if(layer_id == 1) printf("%f-%d\n", tmpin.data, i);
	}
}

void write_dsofm(quant_act ofm[OUTDS_FM_MEM_SIZE], hls::stream<strmio_t> &strm_out) {
	strmio_t tmpout;
	//Write output fm to stream
	for(int i = 0; i < OUTDS_FM_MEM_SIZE; i++){
		if(i == OUTDS_FM_MEM_SIZE - 1) tmpout.last = 1;
		else tmpout.last = 0;
//		printf("sending: %d\n", (int)ofm[i]);
		tmpout.data = ofm[i];
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		strm_out.write(tmpout);
	}
}

void write_ofm(act_reshp ofm[OUTPUT_MEM_SIZE/RESHP_FACTOR], hls::stream<strmio_t> &strm_out) {
	strmio_t tmpout;
	//Write output fm to stream
	for(int i = 0; i < OUTPUT_MEM_SIZE/RESHP_FACTOR; i++){
#pragma HLS PIPELINE II=8
		tmpout.data = (quant_act)ofm[i].range(7,0);
//		printf("%d: %d   \n", (int)(quant_act)ofm[i].range(7,0), (int)tmpout.data);
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		tmpout.last = 0;
		strm_out.write(tmpout);
		tmpout.data = (quant_act)ofm[i].range(15,8);
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		tmpout.last = 0;
		strm_out.write(tmpout);
		tmpout.data = (quant_act)ofm[i].range(23,16);
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		tmpout.last = 0;
		strm_out.write(tmpout);
		tmpout.data = (quant_act)ofm[i].range(31,24);
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		tmpout.last = 0;
		strm_out.write(tmpout);
		tmpout.data = (quant_act)ofm[i].range(39,32);
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		tmpout.last = 0;
		strm_out.write(tmpout);
		tmpout.data = (quant_act)ofm[i].range(47,40);
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		tmpout.last = 0;
		strm_out.write(tmpout);
		tmpout.data = (quant_act)ofm[i].range(55,48);
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		tmpout.last = 0;
		strm_out.write(tmpout);
		if(i == (OUTPUT_MEM_SIZE/RESHP_FACTOR) - 1) tmpout.last = 1;
		else tmpout.last = 0;
		tmpout.data = (quant_act)ofm[i].range(63,56);
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		strm_out.write(tmpout);
	}
}


template<params_t fm_width, params_t fm_height, params_t output_width, params_t output_height, params_t nbands, params_t kernel_size>
void average_pool(act_reshp in_feature_map[fm_width*fm_height*nbands/RESHP_FACTOR], act_reshp out_feature_map[output_height*output_width*nbands/RESHP_FACTOR]){
	ap_int<32> accum = 0;
	quant_act avg = 0;
	ap_uint<3> div = kernel_size*kernel_size;
	int in_idx = 0, out_idx = 0;
	quant_act pixel = 0;

	for(int z = 0; z < nbands; z++){
		for (int i = 0; i < fm_width-1; i+=2) {
			for (int j = 0; j < fm_height-1; j+=2) {
				for(int k = 0; k < kernel_size; k++){
					for(int l = 0; l < kernel_size; l++){
#pragma HLS PIPELINE II=8
						in_idx = ((i+k)*nbands*fm_height + (j+l)*nbands + z);
						if(in_idx % RESHP_FACTOR == 0) pixel = in_feature_map[in_idx/RESHP_FACTOR].range(7,0);
						else if(in_idx % RESHP_FACTOR == 1) pixel = in_feature_map[in_idx/RESHP_FACTOR].range(15,8);
						else if(in_idx % RESHP_FACTOR == 2) pixel = in_feature_map[in_idx/RESHP_FACTOR].range(23,16);
						else if(in_idx % RESHP_FACTOR == 3) pixel = in_feature_map[in_idx/RESHP_FACTOR].range(31,24);
						else if(in_idx % RESHP_FACTOR == 4) pixel = in_feature_map[in_idx/RESHP_FACTOR].range(39,32);
						else if(in_idx % RESHP_FACTOR == 5) pixel = in_feature_map[in_idx/RESHP_FACTOR].range(47,40);
						else if(in_idx % RESHP_FACTOR == 6) pixel = in_feature_map[in_idx/RESHP_FACTOR].range(55,48);
						else if(in_idx % RESHP_FACTOR == 7) pixel = in_feature_map[in_idx/RESHP_FACTOR].range(63,56);

						accum += pixel;
//						printf("accum: %d\n", (int)accum);
						if(k == kernel_size-1 && l == kernel_size-1){
							avg = accum / div;
							out_idx = (i/2)*nbands*output_height+ (j/2)*nbands + z;
							if(out_idx % RESHP_FACTOR == 0) out_feature_map[out_idx/RESHP_FACTOR].range(7,0) = (quant_act) avg;
							else if(out_idx % RESHP_FACTOR == 1) out_feature_map[out_idx/RESHP_FACTOR].range(15,8) = (quant_act) avg;
							else if(out_idx % RESHP_FACTOR == 2) out_feature_map[out_idx/RESHP_FACTOR].range(23,16) = (quant_act) avg;
							else if(out_idx % RESHP_FACTOR == 3) out_feature_map[out_idx/RESHP_FACTOR].range(31,24) = (quant_act) avg;
							else if(out_idx % RESHP_FACTOR == 4) out_feature_map[out_idx/RESHP_FACTOR].range(39,32) = (quant_act) avg;
							else if(out_idx % RESHP_FACTOR == 5) out_feature_map[out_idx/RESHP_FACTOR].range(47,40) = (quant_act) avg;
							else if(out_idx % RESHP_FACTOR == 6) out_feature_map[out_idx/RESHP_FACTOR].range(55,48) = (quant_act) avg;
							else if(out_idx % RESHP_FACTOR == 7) out_feature_map[out_idx/RESHP_FACTOR].range(63,56) = (quant_act) avg;
							accum = 0;
						}
					}
				}
			}
		}
	}
}

template<params_t fm_width, params_t fm_height, params_t nbands_conv, params_t nbands_shortcut>
void add_shortcut(act_reshp conv_feature_map[fm_width*fm_height*nbands_conv/RESHP_FACTOR], act_reshp shortcut[fm_width*fm_height*nbands_shortcut/RESHP_FACTOR], act_reshp out_feature_map[fm_width*fm_height*nbands_conv/RESHP_FACTOR]){
	int idx_conv = 0;
	int idx_shortcut = 0;
	quant_accum sum = 0;
	for (int i = 0; i < fm_width; i++) {
		for (int j = 0; j < fm_height; j++) {
			for(int z = 0; z < nbands_conv/RESHP_FACTOR; z++){
#pragma HLS PIPELINE II=8
				if(z >= nbands_shortcut/RESHP_FACTOR){
					sum = (quant_act)conv_feature_map[idx_conv].range(7,0);
					out_feature_map[idx_conv].range(7,0) = (quant_act) sum;
					sum = (quant_act)conv_feature_map[idx_conv].range(15,8);
					out_feature_map[idx_conv].range(15,8) = (quant_act) sum;
					sum = (quant_act)conv_feature_map[idx_conv].range(23,16);
					out_feature_map[idx_conv].range(23,16) = (quant_act) sum;
					sum = (quant_act)conv_feature_map[idx_conv].range(31,24);
					out_feature_map[idx_conv].range(31,24) = (quant_act) sum;
					sum = (quant_act)conv_feature_map[idx_conv].range(39,32);
					out_feature_map[idx_conv].range(39,32) = (quant_act) sum;
					sum = (quant_act)conv_feature_map[idx_conv].range(47,40);
					out_feature_map[idx_conv].range(47,40) = (quant_act) sum;
					sum = (quant_act)conv_feature_map[idx_conv].range(55,48);
					out_feature_map[idx_conv].range(55,48) = (quant_act) sum;
					sum = (quant_act)conv_feature_map[idx_conv].range(63,56);
					out_feature_map[idx_conv].range(63,56) = (quant_act) sum;
					idx_conv++;
				}
				else {
					sum = (quant_act)conv_feature_map[idx_conv].range(7,0) + shortcut[idx_shortcut].range(7,0);
					out_feature_map[idx_conv].range(7,0) = (quant_act) sum;
//					printf("ds out: %d\n",shortcut[idx_shortcut]);
					sum = (quant_act)conv_feature_map[idx_conv].range(15,8) + shortcut[idx_shortcut].range(15,8);
					out_feature_map[idx_conv].range(15,8) = (quant_act) sum;
					sum = (quant_act)conv_feature_map[idx_conv].range(23,16) + shortcut[idx_shortcut].range(23,16);
					out_feature_map[idx_conv].range(23,16) = (quant_act) sum;
					sum = (quant_act)conv_feature_map[idx_conv].range(31,24) + shortcut[idx_shortcut].range(31,24);
					out_feature_map[idx_conv].range(31,24) = (quant_act) sum;
					sum = (quant_act)conv_feature_map[idx_conv].range(39,32) + shortcut[idx_shortcut].range(39,32);
					out_feature_map[idx_conv].range(39,32) = (quant_act) sum;
					sum = (quant_act)conv_feature_map[idx_conv].range(47,40) + shortcut[idx_shortcut].range(47,40);
					out_feature_map[idx_conv].range(47,40) = (quant_act) sum;
					sum = (quant_act)conv_feature_map[idx_conv].range(55,48) + shortcut[idx_shortcut].range(55,48);
					out_feature_map[idx_conv].range(55,48) = (quant_act) sum;
					sum = (quant_act)conv_feature_map[idx_conv].range(63,56) + shortcut[idx_shortcut].range(63,56);
					out_feature_map[idx_conv].range(63,56) = (quant_act) sum;
					idx_shortcut++;
					idx_conv++;
				}
			}
		}
	}
}

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, wght_reshp *weights, params_t PE, bool relu>
void conv_layer_k1(act_reshp in_feature_map[fm_height*fm_width*nbands/RESHP_FACTOR], act_reshp out_feature_map[fm_height*fm_width*nfilters/RESHP_FACTOR]) {
	quant_accum acc = 0;
	int kernel_idx = 0;
	int input_idx = 0;
	int output_idx = 0;
	ap_uint<3> range_idx = 0;
	act_reshp acc_tmp = 0;
	quant_act tmp_out = 0;
	act_reshp tmp_in = 0;
	wght_reshp tmp_weight = 0;


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
					for(int p = 0; p < PE/RESHP_FACTOR; p++) {
						tmp_in = in_feature_map[input_idx];
						tmp_weight = weights[kernel_idx];
						acc += (quant_wght)tmp_weight.range(1,0) * (quant_act)tmp_in.range(7,0);
						acc += (quant_wght)tmp_weight.range(3,2) * (quant_act)tmp_in.range(15,8);
						acc += (quant_wght)tmp_weight.range(5,4) * (quant_act)tmp_in.range(23,16);
						acc += (quant_wght)tmp_weight.range(7,6) * (quant_act)tmp_in.range(31,24);
						acc += (quant_wght)tmp_weight.range(9,8) * (quant_act)tmp_in.range(39,32);
						acc += (quant_wght)tmp_weight.range(11,10) * (quant_act)tmp_in.range(47,40);
						acc += (quant_wght)tmp_weight.range(13,12) * (quant_act)tmp_in.range(55,48);
						acc += (quant_wght)tmp_weight.range(15,14) * (quant_act)tmp_in.range(63,56);
						kernel_idx++;
						input_idx++;
						if(z + (p*RESHP_FACTOR) == nbands-RESHP_FACTOR) {
							tmp_out = (quant_act)acc;
							if(relu) tmp_out = (ap_int<1>)((quant_act)acc[7]) == 0 ? tmp_out : (quant_act)0;
							if(k%RESHP_FACTOR==0) acc_tmp.range(7,0) = tmp_out;
							else if(k%RESHP_FACTOR==1) acc_tmp.range(15,8) = tmp_out;
							else if(k%RESHP_FACTOR==2) acc_tmp.range(23,16) = tmp_out;
							else if(k%RESHP_FACTOR==3) acc_tmp.range(31,24) = tmp_out;
							else if(k%RESHP_FACTOR==4) acc_tmp.range(39,32) = tmp_out;
							else if(k%RESHP_FACTOR==5) acc_tmp.range(47,40) = tmp_out;
							else if(k%RESHP_FACTOR==6) acc_tmp.range(55,48) = tmp_out;
							else {
								acc_tmp.range(63,56) = tmp_out;
								out_feature_map[output_idx] = acc_tmp; //ver o fator de escala aqui
//								printf("acc_tmp: %ld\n", (long)acc_tmp);
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
template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, wght_reshp *weights, params_t PE, bool relu>
void conv_layer_k1_b4k2_x9(act_reshp in_feature_map[fm_height*fm_width*nbands/RESHP_FACTOR], act_reshp out_feature_map[2][(fm_height-1)*(fm_width-1)*nfilters/2/RESHP_FACTOR]) {
	quant_accum acc = 0;
	int kernel_idx = 0;
	int input_idx = 0;
	int output_idx_even = 0;
	int output_idx_odd = 0;
	ap_uint<3> range_idx = 0;
	act_reshp acc_tmp = 0;
	act_reshp tmp_in = 0;
	wght_reshp tmp_weight = 0;
	quant_act tmp_out = 0;

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
					for(int p = 0; p < PE/RESHP_FACTOR; p++) {
						if(j == fm_height-1){
//							printf("J MAIOR\n");
							if(k == 0 && z == 0 && p == 0)
								input_idx += nbands/RESHP_FACTOR;
	//						z = nbands;
	//						k = nfilters;
						}
						else {
							tmp_in = in_feature_map[input_idx];
							tmp_weight = weights[kernel_idx];
							acc += (quant_wght)tmp_weight.range(1,0) * (quant_act)tmp_in.range(7,0);
							acc += (quant_wght)tmp_weight.range(3,2) * (quant_act)tmp_in.range(15,8);
							acc += (quant_wght)tmp_weight.range(5,4) * (quant_act)tmp_in.range(23,16);
							acc += (quant_wght)tmp_weight.range(7,6) * (quant_act)tmp_in.range(31,24);
							acc += (quant_wght)tmp_weight.range(9,8) * (quant_act)tmp_in.range(39,32);
							acc += (quant_wght)tmp_weight.range(11,10) * (quant_act)tmp_in.range(47,40);
							acc += (quant_wght)tmp_weight.range(13,12) * (quant_act)tmp_in.range(55,48);
							acc += (quant_wght)tmp_weight.range(15,14) * (quant_act)tmp_in.range(63,56);
							kernel_idx++;
							input_idx++;
							if(z + (p*RESHP_FACTOR) == nbands-RESHP_FACTOR) {
								tmp_out = (quant_act)acc;
								if(relu) tmp_out = (ap_int<1>)((quant_act)acc[7]) == 0 ? tmp_out : (quant_act)0;
//								printf("tmp_out: %d\n", (int)tmp_out);
								if(k%RESHP_FACTOR==0) acc_tmp.range(7,0) = tmp_out;
								else if(k%RESHP_FACTOR==1) acc_tmp.range(15,8) = tmp_out;
								else if(k%RESHP_FACTOR==2) acc_tmp.range(23,16) = tmp_out;
								else if(k%RESHP_FACTOR==3) acc_tmp.range(31,24) = tmp_out;
								else if(k%RESHP_FACTOR==4) acc_tmp.range(39,32) = tmp_out;
								else if(k%RESHP_FACTOR==5) acc_tmp.range(47,40) = tmp_out;
								else if(k%RESHP_FACTOR==6) acc_tmp.range(55,48) = tmp_out;
								else {
									acc_tmp.range(63,56) = tmp_out;
//									printf("acc_tmp k1b4k2: %ld\n", (long)acc_tmp);
									if(i % 2 == 0) {
										out_feature_map[0][output_idx_even] = acc_tmp; //ver o fator de escala aqui
										output_idx_even++;
									}
		//								printf("index: [0][%d] - %d\n", output_idx_even, (int)out_feature_map[0][output_idx_even]);
		//								printf("index: [0][%d] - %d\n", output_idx_even, (int)acc);
									else {
										out_feature_map[1][output_idx_odd] = acc_tmp; //ver o fator de escala aqui
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

//Convolutional layer to be aplied before a convolution with kernel and stride 2, that will clear the last row and column from the outputs
template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, wght_reshp *weights, params_t PE, bool relu>
void conv_layer_k1_b4k2_x4(act_reshp in_feature_map[fm_height*fm_width*nbands/RESHP_FACTOR], act_reshp out_feature_map[2][(fm_height)*(fm_width)*nfilters/2/RESHP_FACTOR]) {
	quant_accum acc = 0;
	int kernel_idx = 0;
	int input_idx = 0;
	int output_idx_even = 0;
	int output_idx_odd = 0;
	ap_uint<3> range_idx = 0;
	act_reshp acc_tmp = 0;
	act_reshp tmp_in = 0;
	wght_reshp tmp_weight = 0;
	quant_act tmp_out = 0;

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
					for(int p = 0; p < PE/RESHP_FACTOR; p++) {
						tmp_in = in_feature_map[input_idx];
						tmp_weight = weights[kernel_idx];
						acc += (quant_wght)tmp_weight.range(1,0) * (quant_act)tmp_in.range(7,0);
						acc += (quant_wght)tmp_weight.range(3,2) * (quant_act)tmp_in.range(15,8);
						acc += (quant_wght)tmp_weight.range(5,4) * (quant_act)tmp_in.range(23,16);
						acc += (quant_wght)tmp_weight.range(7,6) * (quant_act)tmp_in.range(31,24);
						acc += (quant_wght)tmp_weight.range(9,8) * (quant_act)tmp_in.range(39,32);
						acc += (quant_wght)tmp_weight.range(11,10) * (quant_act)tmp_in.range(47,40);
						acc += (quant_wght)tmp_weight.range(13,12) * (quant_act)tmp_in.range(55,48);
						acc += (quant_wght)tmp_weight.range(15,14) * (quant_act)tmp_in.range(63,56);
						kernel_idx++;
						input_idx++;
						if(z + (p*RESHP_FACTOR) == nbands-RESHP_FACTOR) {
							tmp_out = (quant_act)acc;
							if(relu) tmp_out = (ap_int<1>)((quant_act)acc[7]) == 0 ? tmp_out : (quant_act)0;
							if(k%RESHP_FACTOR==0) acc_tmp.range(7,0) = tmp_out;
							else if(k%RESHP_FACTOR==1) acc_tmp.range(15,8) = tmp_out;
							else if(k%RESHP_FACTOR==2) acc_tmp.range(23,16) = tmp_out;
							else if(k%RESHP_FACTOR==3) acc_tmp.range(31,24) = tmp_out;
							else if(k%RESHP_FACTOR==4) acc_tmp.range(39,32) = tmp_out;
							else if(k%RESHP_FACTOR==5) acc_tmp.range(47,40) = tmp_out;
							else if(k%RESHP_FACTOR==6) acc_tmp.range(55,48) = tmp_out;
							else {
								acc_tmp.range(63,56) = tmp_out;

								if(i % 2 == 0) {
									out_feature_map[0][output_idx_even] = acc_tmp; //ver o fator de escala aqui
									output_idx_even++;
								}
	//								printf("index: [0][%d] - %d\n", output_idx_even, (int)out_feature_map[0][output_idx_even]);
	//								printf("index: [0][%d] - %d\n", output_idx_even, (int)acc);
								else {
									out_feature_map[1][output_idx_odd] = acc_tmp; //ver o fator de escala aqui
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
	return;
}
//Convolution with stride 2
template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t output_dim,wght_reshp *weights, params_t PE, bool relu>
void conv_layer_k2(act_reshp in_feature_map[2][fm_height*fm_width*nbands/2/RESHP_FACTOR], act_reshp out_feature_map[(fm_height/2)*(fm_width/2)*nfilters/RESHP_FACTOR]) {
	quant_accum acc_even = 0, acc_odd = 0;
	ap_uint<2> kernel_size = 2;
	int kernel_idx = 0;
	int input_idx = 0;
	int output_idx = 0;
	ap_uint<3> range_idx = 0;
	act_reshp acc_tmp = 0;
	quant_act tmp_out = 0;
	act_reshp tmp_in0 = 0, tmp_in1 = 0;
	wght_reshp tmp_weight = 0;

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
					for(int p = 0; p < PE/RESHP_FACTOR; p++) {
						tmp_in0 = in_feature_map[0][input_idx];
						tmp_in1 = in_feature_map[1][input_idx];
						tmp_weight = weights[kernel_idx];
						acc_even += (quant_wght)tmp_weight.range(1,0) * (quant_act)tmp_in0.range(7,0);
						acc_odd += (quant_wght)tmp_weight.range(3,2) * (quant_act)tmp_in1.range(7,0);
						acc_even += (quant_wght)tmp_weight.range(5,4) * (quant_act)tmp_in0.range(15,8);
						acc_odd += (quant_wght)tmp_weight.range(7,6) * (quant_act)tmp_in1.range(15,8);
						acc_even += (quant_wght)tmp_weight.range(9,8) * (quant_act)tmp_in0.range(23,16);
						acc_odd += (quant_wght)tmp_weight.range(11,10) * (quant_act)tmp_in1.range(23,16);
						acc_even += (quant_wght)tmp_weight.range(13,12) * (quant_act)tmp_in0.range(31,24);
						acc_odd += (quant_wght)tmp_weight.range(15,14) * (quant_act)tmp_in1.range(31,24);
						kernel_idx++;
						tmp_weight = weights[kernel_idx];
						acc_even += (quant_wght)tmp_weight.range(1,0) * (quant_act)tmp_in0.range(39,32);
						acc_odd += (quant_wght)tmp_weight.range(3,2) * (quant_act)tmp_in1.range(39,32);
						acc_even += (quant_wght)tmp_weight.range(5,4) * (quant_act)tmp_in0.range(47,40);
						acc_odd += (quant_wght)tmp_weight.range(7,6) * (quant_act)tmp_in1.range(47,40);
						acc_even += (quant_wght)tmp_weight.range(9,8) * (quant_act)tmp_in0.range(55,48);
						acc_odd += (quant_wght)tmp_weight.range(11,10) * (quant_act)tmp_in1.range(55,48);
						acc_even += (quant_wght)tmp_weight.range(13,12) * (quant_act)tmp_in0.range(63,56);
						acc_odd += (quant_wght)tmp_weight.range(15,14) * (quant_act)tmp_in1.range(63,56);
						kernel_idx++;
						input_idx++;

						if(z + (p*RESHP_FACTOR) == nbands*2-RESHP_FACTOR) {
							acc_even += acc_odd;
							tmp_out = (quant_act)acc_even;
							if(relu) tmp_out = (ap_int<1>)((quant_act)acc_even[7]) == 0 ? tmp_out : (quant_act)0;
							if(k%RESHP_FACTOR==0) acc_tmp.range(7,0) = tmp_out;
							else if(k%RESHP_FACTOR==1) acc_tmp.range(15,8) = tmp_out;
							else if(k%RESHP_FACTOR==2) acc_tmp.range(23,16) = tmp_out;
							else if(k%RESHP_FACTOR==3) acc_tmp.range(31,24) = tmp_out;
							else if(k%RESHP_FACTOR==4) acc_tmp.range(39,32) = tmp_out;
							else if(k%RESHP_FACTOR==5) acc_tmp.range(47,40) = tmp_out;
							else if(k%RESHP_FACTOR==6) acc_tmp.range(55,48) = tmp_out;
							else {
								acc_tmp.range(63,56) = tmp_out;
								out_feature_map[output_idx] = acc_tmp; //ver o fator de escala aqui
								output_idx++;
							}
							if(k != nfilters-1) input_idx -= nbands*2/RESHP_FACTOR;
							acc_even = 0;
							acc_odd = 0;
							break;
						}
					}
				}
			}
		}
	}
	return;
}

template<params_t input_size, params_t nfilters, wght_reshp *weights, quant_act *bias>
void fully_connected(act_reshp input_fm[input_size/RESHP_FACTOR], act_reshp output_fm[nfilters/RESHP_FACTOR]) {
	quant_accum acc = 0;
	act_reshp tmp_in = 0, tmp_acc = 0;
	wght_reshp tmp_weight = 0;
	int kernel_idx = 0, output_idx = 0;
	ap_int<9> tmp_out = 0;

	for(int i = 0; i < nfilters; i++) {
		for(int j = 0; j < input_size/RESHP_FACTOR; j++) {
			tmp_in = input_fm[j];
			tmp_weight = weights[kernel_idx];
			acc += (quant_wght)tmp_weight.range(1,0) * (quant_act)tmp_in.range(7,0);
			acc += (quant_wght)tmp_weight.range(3,2) * (quant_act)tmp_in.range(15,8);
			acc += (quant_wght)tmp_weight.range(5,4) * (quant_act)tmp_in.range(23,16);
			acc += (quant_wght)tmp_weight.range(7,6) * (quant_act)tmp_in.range(31,24);
			acc += (quant_wght)tmp_weight.range(9,8) * (quant_act)tmp_in.range(39,32);
			acc += (quant_wght)tmp_weight.range(11,10) * (quant_act)tmp_in.range(47,40);
			acc += (quant_wght)tmp_weight.range(13,12) * (quant_act)tmp_in.range(55,48);
			acc += (quant_wght)tmp_weight.range(15,14) * (quant_act)tmp_in.range(63,56);
			kernel_idx++;

			if(j == input_size/RESHP_FACTOR-1) {
				tmp_out = (quant_act)acc + bias[i];
				if(i%RESHP_FACTOR == 0) tmp_acc.range(7,0) = (quant_act)tmp_out;
				else if(i%RESHP_FACTOR == 1) tmp_acc.range(15,8) = (quant_act)tmp_out;
				else if(i%RESHP_FACTOR == 2) tmp_acc.range(23,16) = (quant_act)tmp_out;
				else if(i%RESHP_FACTOR == 3) tmp_acc.range(31,24) = (quant_act)tmp_out;
				else if(i%RESHP_FACTOR == 4) tmp_acc.range(39,32) = (quant_act)tmp_out;
				else if(i%RESHP_FACTOR == 5) tmp_acc.range(47,40) = (quant_act)tmp_out;
				else if(i%RESHP_FACTOR == 6) tmp_acc.range(55,48) = (quant_act)tmp_out;
				else {
					tmp_acc.range(63,56) = (quant_act) tmp_out;
					output_fm[output_idx] = tmp_acc;
					output_idx++;
				}
				acc = 0;
			}
		}
	}
}

