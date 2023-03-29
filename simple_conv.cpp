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
//#include "weights_reshaped_7.h"    //Weights reshaped for 8x2 quantization with a factor of 8 with 64 bands and 168 filters on the last layer
//#include "weights_reshaped_8.h"    //Weights reshaped with fully connected layer for 8x2 quantization with a factor of 8 with 64 bands and 168 filters on the last layer
//#include "weights_reshaped_9.h"    //Weights reshaped with fully connected layer for 8x2 quantization with a factor of 8 with 64 bands and 168 filters on the last layer
//#include "weights_reshaped_6.h"    //Weights reshaped for 8x2 quantization with a factor of 8 with 64 bands and 168 filters on the last layer
#include "weights.h"


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

	const params_t sf_i = 1, sf_weights = 1, sf_o = 2, sf_bias = 2, sf_conv = 2, sf_shortcut = 1;

#pragma HLS DATAFLOW
	read_ifm(strm_in, in_feature_map, shortcut_fm);

	average_pool<X1,Y1,X13,Y13,Z1,2,sf_i,sf_o>(shortcut_fm, ds_ofm);
//	write_dsofm(ds_ofm, strm_out);
	conv_layer_k1_b4k2_x4<0,X1,Y1,Z1,NF1,sf_i,sf_weights,sf_o, weights_l1, 8, false> (in_feature_map, m1_feature_map);
	conv_layer_k2<1,X2,Y2,Z2,NF2, X3,sf_i,sf_weights,sf_o, weights_l2, 8, false> (m1_feature_map, m2_feature_map);
	conv_layer_k1<2,X3,Y3,Z3,NF3,sf_i,sf_weights,sf_o, weights_l3,8, false> (m2_feature_map, m3_feature_map);
	add_shortcut<X3,Y3,NF3,Z13,sf_conv,sf_shortcut,sf_o>(m3_feature_map, ds_ofm, m4_feature_map);
	average_pool<X3,Y3,X13,Y13,NF3,K13,sf_i,sf_o>(m4_feature_map, m5_feature_map);
	fully_connected<NF3,NCLASSES, weights_fc, bias_fc, sf_i,sf_weights,sf_bias,sf_o>(m5_feature_map, out_feature_map);


//	conv_layer_k1<0,X1,Y1,Z1,NF1, weights_l1,8,false> (in_feature_map, l1_fm);
//
//	conv_layer_k1<1,X2,Y2,Z2,NF2, weights_l2,8,true> (l1_fm, l2_fm);
//	conv_layer_k1<2,X3,Y3,Z3,NF3, weights_l3,8,true> (l2_fm, l3_fm);
//	conv_layer_k1<3,X4,Y4,Z4,NF4, weights_l4,8,false> (l3_fm, l4_fm);
//
//	conv_layer_k1<4,X2,Y2,Z2,NF2, weights_l2,8,true> (l4_fm, l5_fm);
//	conv_layer_k1<5,X3,Y3,Z3,NF3, weights_l3,8,true> (l5_fm, l6_fm);
//	conv_layer_k1<6,X4,Y4,Z4,NF4, weights_l4,8,false> (l6_fm, l7_fm);
//
//	conv_layer_k1<7,X2,Y2,Z2,NF2, weights_l2,8,true> (l7_fm, l8_fm);
//	conv_layer_k1<8,X3,Y3,Z3,NF3, weights_l3,8,true> (l8_fm, l9_fm);
//	conv_layer_k1<9,X4,Y4,Z4,NF4, weights_l4,8,false> (l9_fm, l10_fm);
//
//
//
//
//
//	average_pool<X1,Y1,XDS,YDS,Z1,KDS>(shortcut_fm, ds_ofm);
//	conv_layer_k1_b4k2<0,X1,Y1,Z1,NF1, weights_l1, 24> (in_feature_map, m1_feature_map);
//	conv_layer_k2<1,X2-1,Y2-1,Z2,NF2, X3, weights_l2, 8> (m1_feature_map, m2_feature_map);
//	conv_layer_k1<2,X3,Y3,Z3,NF3, weights_l3,32,true> (m2_feature_map, m3_feature_map);
//	add_shortcut<X3,Y3,NF3,ZDS>(m3_feature_map, ds_ofm, out_feature_map);

	write_ofm(out_feature_map, strm_out);
}


void read_ifm(hls::stream<strmio_t> &strm_in, act_reshp in_feature_map[INPUT1_MEM_SIZE/RESHP_FACTOR], act_reshp shortcut_ifm[INPUT1_MEM_SIZE/RESHP_FACTOR]){
	strmio_t tmpin;
	act_reshp tmp = 0;

	//read input fm
	for(int i = 0; i < INPUT1_MEM_SIZE/RESHP_FACTOR; i++) {
#pragma HLS PIPELINE II=8
		tmpin = strm_in.read();
		tmp.range(3,0) = tmpin.data;
//		printf("%d   ", (int)(quant_act)(in_feature_map[i].range(3,0)));
		if(tmpin.last == 1) break;
		tmpin = strm_in.read();
		tmp.range(7,4) = tmpin.data;
//		printf("%d: %d   ", (int)tmpin.data, (int)((quant_act)in_feature_map[i].range(7,4)));
		if(tmpin.last == 1) break;
		tmpin = strm_in.read();
		tmp.range(11,8) = tmpin.data;
//		printf("%d: %d   ", (int)tmpin.data, (int)((quant_act)in_feature_map[i].range(11,8)));
		if(tmpin.last == 1) break;
		tmpin = strm_in.read();
		tmp.range(15,12) = tmpin.data;
		if(tmpin.last == 1) break;
		tmpin = strm_in.read();
		tmp.range(19,16) = tmpin.data;
//		printf("%d: %d   ", (int)tmpin.data, (int)((quant_act)in_feature_map[i].range(11,8)));
		if(tmpin.last == 1) break;
		tmpin = strm_in.read();
		tmp.range(23,20) = tmpin.data;
		//printf("%d: %d   ", (int)tmpin.data, (int)((quant_act)in_feature_map[i].range(11,8)));
		if(tmpin.last == 1) break;
		tmpin = strm_in.read();
		tmp.range(27,24) = tmpin.data;
		//printf("%d: %d   ", (int)tmpin.data, (int)((quant_act)in_feature_map[i].range(11,8)));
		if(tmpin.last == 1) break;
		tmpin = strm_in.read();
		tmp.range(31,28) = tmpin.data;
		in_feature_map[i] = tmp;
		shortcut_ifm[i] = tmp;
		//printf("%d: %d   ", (int)tmpin.data, (int)((quant_act)in_feature_map[i].range(11,8)));
		if(tmpin.last == 1) break;

//		if(layer_id == 1) printf("%f-%d\n", tmpin.data, i);
	}
}

void write_ofm(act_reshp ofm[NCLASSES/RESHP_FACTOR], hls::stream<strmio_t> &strm_out) {
	strmio_t tmpout;
	//Write output fm to stream
	for(int i = 0; i < NCLASSES/RESHP_FACTOR; i++){
#pragma HLS PIPELINE II=8
		tmpout.data = (quant_act)ofm[i].range(3,0);
//		printf("%d: %d   \n", (int)(quant_act)ofm[i].range(3,0), (int)tmpout.data);
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		tmpout.last = 0;
		strm_out.write(tmpout);
		tmpout.data = (quant_act)ofm[i].range(7,4);
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		tmpout.last = 0;
		strm_out.write(tmpout);
		tmpout.data = (quant_act)ofm[i].range(11,8);
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		tmpout.last = 0;
		strm_out.write(tmpout);
		tmpout.data = (quant_act)ofm[i].range(15,12);
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		tmpout.last = 0;
		strm_out.write(tmpout);
		tmpout.data = (quant_act)ofm[i].range(19,16);
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		tmpout.last = 0;
		strm_out.write(tmpout);
		tmpout.data = (quant_act)ofm[i].range(23,20);
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		tmpout.last = 0;
		strm_out.write(tmpout);
		tmpout.data = (quant_act)ofm[i].range(27,24);
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		tmpout.last = 0;
		strm_out.write(tmpout);
		if(i == (OUTPUT_MEM_SIZE/RESHP_FACTOR) - 1) tmpout.last = 1;
		else tmpout.last = 0;
		tmpout.data = (quant_act)ofm[i].range(31,28);
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		strm_out.write(tmpout);
	}
}


template<params_t fm_width, params_t fm_height, params_t output_width, params_t output_height, params_t nbands, params_t kernel_size, params_t sf_i, params_t sf_o>
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
						if(in_idx % RESHP_FACTOR == 0) pixel = in_feature_map[in_idx/RESHP_FACTOR].range(3,0);
						else if(in_idx % RESHP_FACTOR == 1) pixel = in_feature_map[in_idx/RESHP_FACTOR].range(7,4);
						else if(in_idx % RESHP_FACTOR == 2) pixel = in_feature_map[in_idx/RESHP_FACTOR].range(11,8);
						else if(in_idx % RESHP_FACTOR == 3) pixel = in_feature_map[in_idx/RESHP_FACTOR].range(15,12);
						else if(in_idx % RESHP_FACTOR == 4) pixel = in_feature_map[in_idx/RESHP_FACTOR].range(19,16);
						else if(in_idx % RESHP_FACTOR == 5) pixel = in_feature_map[in_idx/RESHP_FACTOR].range(23,20);
						else if(in_idx % RESHP_FACTOR == 6) pixel = in_feature_map[in_idx/RESHP_FACTOR].range(27,24);
						else if(in_idx % RESHP_FACTOR == 7) pixel = in_feature_map[in_idx/RESHP_FACTOR].range(31,28);

						accum += pixel;
//						printf("accum: %d\n", (int)accum);
						if(k == kernel_size-1 && l == kernel_size-1){
							avg = accum / div;
							avg = avg >> sf_i-sf_o;
							avg = avg.range(3,0);
							out_idx = (i/2)*nbands*output_height+ (j/2)*nbands + z;
							if(out_idx % RESHP_FACTOR == 0) out_feature_map[out_idx/RESHP_FACTOR].range(3,0) = avg;
							else if(out_idx % RESHP_FACTOR == 1) out_feature_map[out_idx/RESHP_FACTOR].range(7,4) = avg;
							else if(out_idx % RESHP_FACTOR == 2) out_feature_map[out_idx/RESHP_FACTOR].range(11,8) = avg;
							else if(out_idx % RESHP_FACTOR == 3) out_feature_map[out_idx/RESHP_FACTOR].range(15,12) = avg;
							else if(out_idx % RESHP_FACTOR == 4) out_feature_map[out_idx/RESHP_FACTOR].range(19,16) = avg;
							else if(out_idx % RESHP_FACTOR == 5) out_feature_map[out_idx/RESHP_FACTOR].range(23,20) = avg;
							else if(out_idx % RESHP_FACTOR == 6) out_feature_map[out_idx/RESHP_FACTOR].range(27,24) = avg;
							else if(out_idx % RESHP_FACTOR == 7) out_feature_map[out_idx/RESHP_FACTOR].range(31,28) = avg;
							accum = 0;
						}
					}
				}
			}
		}
	}
}

template<params_t fm_width, params_t fm_height, params_t nbands_conv, params_t nbands_shortcut, params_t sf_conv, params_t sf_shortcut, params_t sf_o>
void add_shortcut(act_reshp conv_feature_map[fm_width*fm_height*nbands_conv/RESHP_FACTOR], act_reshp shortcut[fm_width*fm_height*nbands_shortcut/RESHP_FACTOR], act_reshp out_feature_map[fm_width*fm_height*nbands_conv/RESHP_FACTOR]){
	int idx_conv = 0;
	int idx_shortcut = 0;
	quant_accum sum = 0;
	quant_act tmp_conv = 0, tmp_shortcut = 0;
	params_t scale_factor = sf_conv;
	for (int i = 0; i < fm_width; i++) {
		for (int j = 0; j < fm_height; j++) {
			for(int z = 0; z < nbands_conv; z++){
				if(z%RESHP_FACTOR == 0) tmp_conv = (quant_act)conv_feature_map[idx_conv].range(3,0);
				else if(z%RESHP_FACTOR == 1) tmp_conv = (quant_act)conv_feature_map[idx_conv].range(7,4);
				else if(z%RESHP_FACTOR == 2) tmp_conv = (quant_act)conv_feature_map[idx_conv].range(11,8);
				else if(z%RESHP_FACTOR == 3) tmp_conv = (quant_act)conv_feature_map[idx_conv].range(15,12);
				else if(z%RESHP_FACTOR == 4) tmp_conv = (quant_act)conv_feature_map[idx_conv].range(19,16);
				else if(z%RESHP_FACTOR == 5) tmp_conv = (quant_act)conv_feature_map[idx_conv].range(23,20);
				else if(z%RESHP_FACTOR == 6) tmp_conv = (quant_act)conv_feature_map[idx_conv].range(27,24);
				else if(z%RESHP_FACTOR == 7) tmp_conv = (quant_act)conv_feature_map[idx_conv].range(31,28);

				if(z >= nbands_shortcut){
					sum = tmp_conv;
				}
				else {
					if(z%RESHP_FACTOR == 0) tmp_shortcut = (quant_act)shortcut[idx_shortcut].range(3,0);
					else if(z%RESHP_FACTOR == 1) tmp_shortcut = (quant_act)shortcut[idx_shortcut].range(7,4);
					else if(z%RESHP_FACTOR == 2) tmp_shortcut = (quant_act)shortcut[idx_shortcut].range(11,8);
					else if(z%RESHP_FACTOR == 3) tmp_shortcut = (quant_act)shortcut[idx_shortcut].range(15,12);
					else if(z%RESHP_FACTOR == 4) tmp_shortcut = (quant_act)shortcut[idx_shortcut].range(19,16);
					else if(z%RESHP_FACTOR == 5) tmp_shortcut = (quant_act)shortcut[idx_shortcut].range(23,20);
					else if(z%RESHP_FACTOR == 6) tmp_shortcut = (quant_act)shortcut[idx_shortcut].range(27,24);
					else if(z%RESHP_FACTOR == 7) {
						tmp_shortcut = (quant_act)shortcut[idx_shortcut].range(31,28);
						idx_shortcut++;
					}
					if(sf_conv > sf_shortcut) {
						tmp_shortcut = tmp_shortcut << (sf_conv - sf_shortcut);
						scale_factor = sf_conv;
					} else if( sf_shortcut > sf_conv) {
						tmp_conv = tmp_conv << (sf_shortcut-sf_conv);
						scale_factor = sf_shortcut;
					}
					sum = tmp_conv + tmp_shortcut;
				}

				sum = sum >> scale_factor-sf_o;
				sum = sum.range(3,0);

				if(z%RESHP_FACTOR == 0) out_feature_map[idx_conv].range(3,0) = sum;
				else if(z%RESHP_FACTOR == 1) out_feature_map[idx_conv].range(7,4) = sum;
				else if(z%RESHP_FACTOR == 2) out_feature_map[idx_conv].range(11,8) = sum;
				else if(z%RESHP_FACTOR == 3) out_feature_map[idx_conv].range(15,12) = sum;
				else if(z%RESHP_FACTOR == 4) out_feature_map[idx_conv].range(19,16) = sum;
				else if(z%RESHP_FACTOR == 5) out_feature_map[idx_conv].range(23,20) = sum;
				else if(z%RESHP_FACTOR == 6) out_feature_map[idx_conv].range(27,24) = sum;
				else if(z%RESHP_FACTOR == 7) {
					out_feature_map[idx_conv].range(31,28) = (quant_act) sum;
					idx_conv++;
				}
			}
		}
	}
}

template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t sf_i, params_t sf_weights, params_t sf_o, wght_reshp *weights, params_t PE, bool relu>
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
						acc += (quant_wght)tmp_weight.range(1,0) * (quant_act)tmp_in.range(3,0);
						acc += (quant_wght)tmp_weight.range(3,2) * (quant_act)tmp_in.range(7,4);
						acc += (quant_wght)tmp_weight.range(5,4) * (quant_act)tmp_in.range(11,8);
						acc += (quant_wght)tmp_weight.range(7,6) * (quant_act)tmp_in.range(15,12);
						acc += (quant_wght)tmp_weight.range(9,8) * (quant_act)tmp_in.range(19,16);
						acc += (quant_wght)tmp_weight.range(11,10) * (quant_act)tmp_in.range(23,20);
						acc += (quant_wght)tmp_weight.range(13,12) * (quant_act)tmp_in.range(27,24);
						acc += (quant_wght)tmp_weight.range(15,14) * (quant_act)tmp_in.range(31,28);
						kernel_idx++;
						input_idx++;
						if(z + (p*RESHP_FACTOR) == nbands-RESHP_FACTOR) {
//							tmp_out = (quant_act)acc;
							acc = acc>>((sf_weights+sf_i)-sf_o);
							tmp_out = acc.range(3,0);
//							printf("%X\n\n\n",tmp_out);
							if(relu) tmp_out = (ap_int<1>)((quant_act)acc[7]) == 0 ? tmp_out : (quant_act)0;
							if(k%RESHP_FACTOR==0) acc_tmp.range(3,0) = tmp_out;
							else if(k%RESHP_FACTOR==1) acc_tmp.range(7,4) = tmp_out;
							else if(k%RESHP_FACTOR==2) acc_tmp.range(11,8) = tmp_out;
							else if(k%RESHP_FACTOR==3) acc_tmp.range(15,12) = tmp_out;
							else if(k%RESHP_FACTOR==4) acc_tmp.range(19,16) = tmp_out;
							else if(k%RESHP_FACTOR==5) acc_tmp.range(23,20) = tmp_out;
							else if(k%RESHP_FACTOR==6) acc_tmp.range(27,24) = tmp_out;
							else {
								acc_tmp.range(31,28) = tmp_out;
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
template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t sf_i, params_t sf_weights, params_t sf_o, wght_reshp *weights, params_t PE, bool relu>
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
							acc += (quant_wght)tmp_weight.range(1,0) * (quant_act)tmp_in.range(3,0);
							acc += (quant_wght)tmp_weight.range(3,2) * (quant_act)tmp_in.range(7,4);
							acc += (quant_wght)tmp_weight.range(5,4) * (quant_act)tmp_in.range(11,8);
							acc += (quant_wght)tmp_weight.range(7,6) * (quant_act)tmp_in.range(15,12);
							acc += (quant_wght)tmp_weight.range(9,8) * (quant_act)tmp_in.range(19,16);
							acc += (quant_wght)tmp_weight.range(11,10) * (quant_act)tmp_in.range(23,20);
							acc += (quant_wght)tmp_weight.range(13,12) * (quant_act)tmp_in.range(27,24);
							acc += (quant_wght)tmp_weight.range(15,14) * (quant_act)tmp_in.range(31,28);
							kernel_idx++;
							input_idx++;
							if(z + (p*RESHP_FACTOR) == nbands-RESHP_FACTOR) {
//								tmp_out = (quant_act)acc;
								acc = acc>>((sf_weights+sf_i)-sf_o);
								tmp_out = acc.range(3,0);
								if(relu) tmp_out = (ap_int<1>)((quant_act)acc[7]) == 0 ? tmp_out : (quant_act)0;
//								printf("tmp_out: %d\n", (int)tmp_out);
								if(k%RESHP_FACTOR==0) acc_tmp.range(3,0) = tmp_out;
								else if(k%RESHP_FACTOR==1) acc_tmp.range(7,4) = tmp_out;
								else if(k%RESHP_FACTOR==2) acc_tmp.range(11,8) = tmp_out;
								else if(k%RESHP_FACTOR==3) acc_tmp.range(15,12) = tmp_out;
								else if(k%RESHP_FACTOR==4) acc_tmp.range(19,16) = tmp_out;
								else if(k%RESHP_FACTOR==5) acc_tmp.range(23,20) = tmp_out;
								else if(k%RESHP_FACTOR==6) acc_tmp.range(27,24) = tmp_out;
								else {
									acc_tmp.range(31,28) = tmp_out;
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
template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t sf_i, params_t sf_weights, params_t sf_o, wght_reshp *weights, params_t PE, bool relu>
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
						acc += (quant_wght)tmp_weight.range(1,0) * (quant_act)tmp_in.range(3,0);
						acc += (quant_wght)tmp_weight.range(3,2) * (quant_act)tmp_in.range(7,4);
						acc += (quant_wght)tmp_weight.range(5,4) * (quant_act)tmp_in.range(11,8);
						acc += (quant_wght)tmp_weight.range(7,6) * (quant_act)tmp_in.range(15,12);
						acc += (quant_wght)tmp_weight.range(9,8) * (quant_act)tmp_in.range(19,16);
						acc += (quant_wght)tmp_weight.range(11,10) * (quant_act)tmp_in.range(23,20);
						acc += (quant_wght)tmp_weight.range(13,12) * (quant_act)tmp_in.range(27,24);
						acc += (quant_wght)tmp_weight.range(15,14) * (quant_act)tmp_in.range(31,28);
						kernel_idx++;
						input_idx++;
						if(z + (p*RESHP_FACTOR) == nbands-RESHP_FACTOR) {
//							tmp_out = (quant_act)acc;
//							printf("%X -> ", acc);
							acc = acc>>((sf_weights+sf_i)-sf_o);
							tmp_out = acc.range(3,0);
//							printf("%X -> ", acc);
//							printf("%X\n\n",tmp_out);
							if(relu) tmp_out = (ap_int<1>)((quant_act)acc[7]) == 0 ? tmp_out : (quant_act)0;
							if(k%RESHP_FACTOR==0) acc_tmp.range(3,0) = tmp_out;
							else if(k%RESHP_FACTOR==1) acc_tmp.range(7,4) = tmp_out;
							else if(k%RESHP_FACTOR==2) acc_tmp.range(11,8) = tmp_out;
							else if(k%RESHP_FACTOR==3) acc_tmp.range(15,12) = tmp_out;
							else if(k%RESHP_FACTOR==4) acc_tmp.range(19,16) = tmp_out;
							else if(k%RESHP_FACTOR==5) acc_tmp.range(23,20) = tmp_out;
							else if(k%RESHP_FACTOR==6) acc_tmp.range(27,24) = tmp_out;
							else {
								acc_tmp.range(31,28) = tmp_out;
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
template<params_t layer_id, params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t output_dim, params_t sf_i, params_t sf_weights, params_t sf_o,wght_reshp *weights, params_t PE, bool relu>
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
						acc_even += (quant_wght)tmp_weight.range(1,0) * (quant_act)tmp_in0.range(3,0);
						acc_odd += (quant_wght)tmp_weight.range(3,2) * (quant_act)tmp_in1.range(3,0);
						acc_even += (quant_wght)tmp_weight.range(5,4) * (quant_act)tmp_in0.range(7,4);
						acc_odd += (quant_wght)tmp_weight.range(7,6) * (quant_act)tmp_in1.range(7,4);
						acc_even += (quant_wght)tmp_weight.range(9,8) * (quant_act)tmp_in0.range(11,8);
						acc_odd += (quant_wght)tmp_weight.range(11,10) * (quant_act)tmp_in1.range(11,8);
						acc_even += (quant_wght)tmp_weight.range(13,12) * (quant_act)tmp_in0.range(15,12);
						acc_odd += (quant_wght)tmp_weight.range(15,14) * (quant_act)tmp_in1.range(15,12);
						kernel_idx++;
						tmp_weight = weights[kernel_idx];
						acc_even += (quant_wght)tmp_weight.range(1,0) * (quant_act)tmp_in0.range(19,16);
						acc_odd += (quant_wght)tmp_weight.range(3,2) * (quant_act)tmp_in1.range(19,16);
						acc_even += (quant_wght)tmp_weight.range(5,4) * (quant_act)tmp_in0.range(23,20);
						acc_odd += (quant_wght)tmp_weight.range(7,6) * (quant_act)tmp_in1.range(23,20);
						acc_even += (quant_wght)tmp_weight.range(9,8) * (quant_act)tmp_in0.range(27,24);
						acc_odd += (quant_wght)tmp_weight.range(11,10) * (quant_act)tmp_in1.range(27,24);
						acc_even += (quant_wght)tmp_weight.range(13,12) * (quant_act)tmp_in0.range(31,28);
						acc_odd += (quant_wght)tmp_weight.range(15,14) * (quant_act)tmp_in1.range(31,28);
						kernel_idx++;
						input_idx++;

						if(z + (p*RESHP_FACTOR) == nbands*2-RESHP_FACTOR) {
//							acc_even += acc_odd;
//							tmp_out = (quant_act)acc_even;
							acc_even = acc_even>>((sf_weights+sf_i)-sf_o);
							tmp_out = acc_even.range(3,0);
							if(relu) tmp_out = (ap_int<1>)((quant_act)acc_even[7]) == 0 ? tmp_out : (quant_act)0;
							if(k%RESHP_FACTOR==0) acc_tmp.range(3,0) = tmp_out;
							else if(k%RESHP_FACTOR==1) acc_tmp.range(7,4) = tmp_out;
							else if(k%RESHP_FACTOR==2) acc_tmp.range(11,8) = tmp_out;
							else if(k%RESHP_FACTOR==3) acc_tmp.range(15,12) = tmp_out;
							else if(k%RESHP_FACTOR==4) acc_tmp.range(19,16) = tmp_out;
							else if(k%RESHP_FACTOR==5) acc_tmp.range(23,20) = tmp_out;
							else if(k%RESHP_FACTOR==6) acc_tmp.range(27,24) = tmp_out;
							else {
								acc_tmp.range(31,28) = tmp_out;
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

template<params_t input_size, params_t nfilters, wght_reshp *weights, quant_bias *bias, params_t sf_i, params_t sf_weights, params_t sf_bias, params_t sf_o>
void fully_connected(act_reshp input_fm[input_size/RESHP_FACTOR], act_reshp output_fm[nfilters/RESHP_FACTOR]) {
	quant_accum acc = 0, sum = 0;
	act_reshp tmp_in = 0, tmp_acc = 0;
	wght_reshp tmp_weight = 0;
	int kernel_idx = 0, output_idx = 0, sf = sf_i;
	ap_int<9> tmp_out = 0;
	quant_bias tmp_bias = 0;

	for(int i = 0; i < nfilters; i++) {
		for(int j = 0; j < input_size/RESHP_FACTOR; j++) {
			tmp_in = input_fm[j];
			tmp_weight = weights[kernel_idx];
			acc += (quant_wght)tmp_weight.range(1,0) * (quant_act)tmp_in.range(3,0);
			acc += (quant_wght)tmp_weight.range(3,2) * (quant_act)tmp_in.range(7,4);
			acc += (quant_wght)tmp_weight.range(5,4) * (quant_act)tmp_in.range(11,8);
			acc += (quant_wght)tmp_weight.range(7,6) * (quant_act)tmp_in.range(15,12);
			acc += (quant_wght)tmp_weight.range(9,8) * (quant_act)tmp_in.range(19,16);
			acc += (quant_wght)tmp_weight.range(11,10) * (quant_act)tmp_in.range(23,20);
			acc += (quant_wght)tmp_weight.range(13,12) * (quant_act)tmp_in.range(27,24);
			acc += (quant_wght)tmp_weight.range(15,14) * (quant_act)tmp_in.range(31,28);
			kernel_idx++;

			if(j == input_size/RESHP_FACTOR-1) {
//				acc = acc>>((sf_weights+sf_i)-sf_o);
//				acc = acc.range(3,0);
				tmp_bias = bias[i];
				if(sf_bias > (sf_i+sf_weights)) {
					acc = acc<<(sf_bias-(sf_i+sf_weights));
					sf = sf_bias;
				} else if( (sf_i+sf_weights) > sf_bias) {
					tmp_bias = tmp_bias<<((sf_i+sf_weights)-sf_bias);
					sf = sf_i+sf_weights;
				}
				sum = acc + tmp_bias;
				sum = sum >> sf - sf_o;
				tmp_out = sum.range(3,0);
				if(i%RESHP_FACTOR == 0) tmp_acc.range(3,0) = (quant_act)tmp_out;
				else if(i%RESHP_FACTOR == 1) tmp_acc.range(7,4) = (quant_act)tmp_out;
				else if(i%RESHP_FACTOR == 2) tmp_acc.range(11,8) = (quant_act)tmp_out;
				else if(i%RESHP_FACTOR == 3) tmp_acc.range(15,12) = (quant_act)tmp_out;
				else if(i%RESHP_FACTOR == 4) tmp_acc.range(19,16) = (quant_act)tmp_out;
				else if(i%RESHP_FACTOR == 5) tmp_acc.range(23,20) = (quant_act)tmp_out;
				else if(i%RESHP_FACTOR == 6) tmp_acc.range(27,24) = (quant_act)tmp_out;
				else {
					tmp_acc.range(31,28) = (quant_act) tmp_out;
					output_fm[output_idx] = tmp_acc;
					output_idx++;
				}
				acc = 0;
			}
		}
	}
}

