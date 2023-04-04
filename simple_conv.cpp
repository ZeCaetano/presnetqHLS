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


void simple_conv(hls::stream<strmi_t> &strm_in, hls::stream<strmo_t> &strm_out) {

#pragma HLS INTERFACE axis port=strm_in
#pragma HLS INTERFACE axis port=strm_out

	act_reshp in_feature_map[INPUT1_MEM_SIZE/RESHP_FACTOR];
	act_reshp l1_fm[OUT1_MEM_SIZE/RESHP_FACTOR], l2_fm[OUT2_MEM_SIZE/RESHP_FACTOR], l3_fm[OUT3_MEM_SIZE/RESHP_FACTOR], l4_fm[OUT4_MEM_SIZE/RESHP_FACTOR];
	act_reshp l5_fm[OUT5_MEM_SIZE/RESHP_FACTOR], l6_fm[OUT6_MEM_SIZE/RESHP_FACTOR], l7_fm[OUT7_MEM_SIZE/RESHP_FACTOR], l8_fm[OUT8_MEM_SIZE/RESHP_FACTOR];
	act_reshp l9_fm[OUT9_MEM_SIZE/RESHP_FACTOR], l10_fm[OUT10_MEM_SIZE/RESHP_FACTOR], l11_fm[2][((X12-1)*(Y12-1)*Z12/2)/RESHP_FACTOR], l12_fm[OUT12_MEM_SIZE/RESHP_FACTOR];
	act_reshp l13_fm[OUT13_MEM_SIZE/RESHP_FACTOR], l14_fm[OUT14_MEM_SIZE/RESHP_FACTOR], l15_fm[OUT15_MEM_SIZE/RESHP_FACTOR], l16_fm[OUT16_MEM_SIZE/RESHP_FACTOR];
	act_reshp l17_fm[OUT17_MEM_SIZE/RESHP_FACTOR], l18_fm[OUT18_MEM_SIZE/RESHP_FACTOR], l19_fm[OUT19_MEM_SIZE/RESHP_FACTOR], l20_fm[2][OUT20_MEM_SIZE/2/RESHP_FACTOR];
	act_reshp l21_fm[OUT21_MEM_SIZE/RESHP_FACTOR], l22_fm[OUT22_MEM_SIZE/RESHP_FACTOR], l23_fm[OUT23_MEM_SIZE/RESHP_FACTOR], l24_fm[OUT24_MEM_SIZE/RESHP_FACTOR];
	act_reshp l25_fm[OUT25_MEM_SIZE/RESHP_FACTOR], l26_fm[OUT26_MEM_SIZE/RESHP_FACTOR], l27_fm[OUT27_MEM_SIZE/RESHP_FACTOR], l28_fm[OUT28_MEM_SIZE/RESHP_FACTOR];
	act_reshp sum_1[OUT4_MEM_SIZE/RESHP_FACTOR], sum_2[OUT7_MEM_SIZE/RESHP_FACTOR], sum_3[OUT10_MEM_SIZE/RESHP_FACTOR], sum_4[OUT13_MEM_SIZE/RESHP_FACTOR], sum_5[OUT16_MEM_SIZE/RESHP_FACTOR];
	act_reshp sum_6[OUT19_MEM_SIZE/RESHP_FACTOR], sum_7[OUT22_MEM_SIZE/RESHP_FACTOR], sum_8[OUT25_MEM_SIZE/RESHP_FACTOR], sum_9[OUT28_MEM_SIZE/RESHP_FACTOR];
	act_reshp shortcut_1_fm[INPUT1_MEM_SIZE/RESHP_FACTOR], shortcut_2_fm[INPUT5_MEM_SIZE/RESHP_FACTOR], shortcut_3_fm[INPUT8_MEM_SIZE/RESHP_FACTOR], shortcut_4_fm[INPUT11_MEM_SIZE/RESHP_FACTOR], shortcut_5_fm[INPUT14_MEM_SIZE/RESHP_FACTOR];
	act_reshp shortcut_6_fm[INPUT17_MEM_SIZE/RESHP_FACTOR],shortcut_7_fm[INPUT20_MEM_SIZE/RESHP_FACTOR], shortcut_8_fm[INPUT23_MEM_SIZE/RESHP_FACTOR], shortcut_9_fm[INPUT26_MEM_SIZE/RESHP_FACTOR];
	act_reshp in_blk2_fm[INPUT5_MEM_SIZE/RESHP_FACTOR], in_blk3_fm[INPUT8_MEM_SIZE/RESHP_FACTOR], in_blk4_fm[INPUT11_MEM_SIZE/RESHP_FACTOR], in_blk5_fm[INPUT14_MEM_SIZE/RESHP_FACTOR];
	act_reshp in_blk6_fm[INPUT17_MEM_SIZE/RESHP_FACTOR],in_blk7_fm[INPUT20_MEM_SIZE/RESHP_FACTOR], in_blk8_fm[INPUT23_MEM_SIZE/RESHP_FACTOR], in_blk9_fm[INPUT26_MEM_SIZE/RESHP_FACTOR];
	act_reshp final_relu[OUT28_MEM_SIZE/RESHP_FACTOR];
	act_reshp ds_1_fm[XDS1*YDS1*NF10/RESHP_FACTOR], ds_2_fm[XDS2*XDS3*NF19/RESHP_FACTOR], final_ds[XDS3*YDS3*NF28];
	quant_accum output_fm[NCLASSES];

//#pragma HLS BIND_STORAGE variable=m1_feature_map type=RAM_1P
//#pragma HLS BIND_STORAGE variable=in_feature_map type=RAM_1P
//#pragma HLS BIND_STORAGE variable=m2_feature_map type=RAM_1P
//
//#pragma HLS STREAM variable=in_feature_map type=PIPO depth=2
//#pragma HLS STREAM variable=shortcut_fm type=PIPO depth=2
//#pragma HLS STREAM variable=ds_ofm type=PIPO depth=4
//#pragma HLS STREAM variable=m1_feature_map type=PIPO depth=2
//#pragma HLS STREAM variable=m2_feature_map type=PIPO depth=2
//#pragma HLS STREAM variable=m3_feature_map type=PIPO depth=2
//#pragma HLS STREAM variable=out_feature_map type=PIPO depth=2

	const params_t sf_i = 1, sf_weights = 1, sf_o = 2, sf_bias = 2, sf_conv = 2, sf_shortcut = 1;

#pragma HLS DATAFLOW
	read_ifm(strm_in, in_feature_map, shortcut_1_fm);

//	average_pool<X1,Y1,X13,Y13,Z1,2,sf_i,sf_o>(shortcut_fm, ds_ofm);
////	write_dsofm(ds_ofm, strm_out);
//	conv_layer_k1_b4k2_x4<0,X1,Y1,Z1,NF1,sf_i,sf_weights,sf_o, weights_l1, 8, false> (in_feature_map, m1_feature_map);
//	conv_layer_k2<1,X2,Y2,Z2,NF2, X3,sf_i,sf_weights,sf_o, weights_l2, 8, false> (m1_feature_map, m2_feature_map);
//	conv_layer_k1<2,X3,Y3,Z3,NF3,sf_i,sf_weights,sf_o, weights_l3,8, false> (m2_feature_map, m3_feature_map);
//	add_shortcut<X3,Y3,NF3,Z13,sf_conv,sf_shortcut,sf_o>(m3_feature_map, ds_ofm, m4_feature_map);
//	average_pool<X3,Y3,X13,Y13,NF3,K13,sf_i,sf_o>(m4_feature_map, m5_feature_map);
//	fully_connected<NF3,NCLASSES, weights_fc, bias_fc, sf_i,sf_weights,sf_bias,sf_o>(m5_feature_map, out_feature_map);


	conv_layer_k1<X1,Y1,Z1,NF1,SFI1,SFW1,SFO1, weights_l1,8,false> (in_feature_map, l1_fm);



	conv_layer_k1<X2,Y2,Z2,NF2,SFI2,SFW2,SFO2, weights_l2,8,true> (l1_fm, l2_fm);
	conv_layer_k1<X3,Y3,Z3,NF3,SFI3,SFW3,SFO3, weights_l3,8,true> (l2_fm, l3_fm);
	conv_layer_k1<X4,Y4,Z4,NF4,SFI4,SFW4,SFO4, weights_l4,8,false> (l3_fm, l4_fm);

	add_shortcut<X4,Y4,NF4,Z2,SFO4,SFO2,SFBLK1>(l4_fm, shortcut_1_fm, sum_1);
	gen_shortcut<X4,Y4,NF4>(sum_1, in_blk2_fm, shortcut_2_fm);

	conv_layer_k1<X5,Y5,Z5,NF5,SFI5,SFW5,SFO5, weights_l5,8,true> (in_blk2_fm, l5_fm);
	conv_layer_k1<X6,Y6,Z6,NF6,SFI6,SFW6,SFO6, weights_l6,8,true> (l5_fm, l6_fm);
	conv_layer_k1<X7,Y7,Z7,NF7,SFI7,SFW7,SFO7, weights_l7,8,false> (l6_fm, l7_fm);

	add_shortcut<X7,Y7,NF7,Z5,SFO7,SFO5,SFBLK2>(l7_fm, shortcut_2_fm, sum_2);
	gen_shortcut<X7,Y7,NF7>(sum_2, in_blk3_fm, shortcut_3_fm);

	conv_layer_k1<X8,Y8,Z8,NF8,SFI8,SFW8,SFO8, weights_l8,8,true> (in_blk3_fm, l8_fm);
	conv_layer_k1<X9,Y9,Z9,NF9,SFI9,SFW9,SFO9, weights_l9,8,true> (l8_fm, l9_fm);
	conv_layer_k1<X10,Y10,Z10,NF10,SFI10,SFW10,SFO10, weights_l10,8,false> (l9_fm, l10_fm);

	add_shortcut<X10,Y10,NF10,Z8,SFO10,SFO8,SFBLK3>(l10_fm, shortcut_3_fm, sum_3);
	gen_shortcut<X10,Y10,NF10>(sum_3, in_blk4_fm, shortcut_4_fm);



	average_pool<X10,Y10,XDS1,YDS1,NF10,KDS1,SFBLK3,SFDS1>(shortcut_4_fm, ds_1_fm);

	conv_layer_k1_b4k2_x9<X11,Y11,Z11,NF11,SFI11,SFW11,SFO11, weights_l11, 8, true> (in_blk4_fm, l11_fm);
	conv_layer_k2<X12-1,Y12-1,Z12,NF12,X13,SFI12,SFW12,SFO12, weights_l12, 8, true> (l11_fm, l12_fm);
	conv_layer_k1<X13,Y13,Z13,NF13,SFI13,SFW13,SFO13, weights_l13,8,false> (l12_fm, l13_fm);

	add_shortcut<X13,Y13,NF13,Z11,SFO13,SFO11,SFBLK4>(l13_fm, ds_1_fm, sum_4);
	gen_shortcut<X13,Y13,NF13>(sum_4, in_blk5_fm, shortcut_5_fm);

	conv_layer_k1<X14,Y14,Z14,NF14,SFI14,SFW14,SFO14, weights_l14,8,true> (in_blk5_fm, l14_fm);
	conv_layer_k1<X15,Y15,Z15,NF15,SFI15,SFW15,SFO15, weights_l15,8,true> (l14_fm, l15_fm);
	conv_layer_k1<X16,Y16,Z16,NF16,SFI16,SFW16,SFO16, weights_l16,8,false> (l15_fm, l16_fm);

	add_shortcut<X16,Y16,NF16,Z14,SFO16,SFO14,SFBLK5>(l16_fm, shortcut_5_fm, sum_5);
	gen_shortcut<X16,Y16,NF16>(sum_5, in_blk6_fm, shortcut_6_fm);

	conv_layer_k1<X17,Y17,Z17,NF17,SFI17,SFW17,SFO17, weights_l17,8,true> (in_blk6_fm, l17_fm);
	conv_layer_k1<X18,Y18,Z18,NF18,SFI18,SFW18,SFO18, weights_l18,8,true> (l17_fm, l18_fm);
	conv_layer_k1<X19,Y19,Z19,NF19,SFI19,SFW19,SFO19, weights_l19,8,false> (l18_fm, l19_fm);

	add_shortcut<X19,Y19,NF19,Z17,SFO19,SFO17,SFBLK6>(l19_fm, shortcut_6_fm, sum_6);
	gen_shortcut<X19,Y19,NF19>(sum_6, in_blk7_fm, shortcut_7_fm);
//
//
//
	average_pool<X19,Y19,XDS2,YDS2,NF19,KDS2,SFBLK6,SFDS2>(shortcut_7_fm, ds_2_fm);

	conv_layer_k1_b4k2_x4<X20,Y20,Z20,NF20,SFI20,SFW20,SFO20, weights_l20, 8, true> (in_blk7_fm, l20_fm);
	conv_layer_k2<X21,Y21,Z21,NF21,X22,SFI21,SFW21,SFO21, weights_l21, 8, true> (l20_fm, l21_fm);
	conv_layer_k1<X22,Y2,Z22,NF22,SFI22,SFW22,SFO22, weights_l22,8,false> (l21_fm, l22_fm);

	add_shortcut<X22,Y22,NF2,Z20,SFO22,SFO20,SFBLK7>(l22_fm, ds_2_fm, sum_7);
	gen_shortcut<X22,Y22,NF22>(sum_7, in_blk8_fm, shortcut_8_fm);

	conv_layer_k1<X23,Y23,Z23,NF23,SFI23,SFW23,SFO23, weights_l23,8,true> (in_blk8_fm, l23_fm);
	conv_layer_k1<X24,Y24,Z24,NF24,SFI24,SFW24,SFO24, weights_l24,8,true> (l23_fm, l24_fm);
	conv_layer_k1<X25,Y25,Z25,NF25,SFI25,SFW25,SFO25, weights_l25,8,false> (l24_fm, l25_fm);

	add_shortcut<X25,Y25,NF25,Z23,SFO25,SFO23,SFBLK8>(l25_fm, shortcut_8_fm, sum_8);
	gen_shortcut<X25,Y25,NF25>(sum_8, in_blk9_fm, shortcut_9_fm);

	conv_layer_k1<X26,Y26,Z26,NF26,SFI26,SFW26,SFO26, weights_l26,8,true> (in_blk9_fm, l26_fm);
	conv_layer_k1<X27,Y27,Z27,NF27,SFI27,SFW27,SFO27, weights_l27,8,true> (l26_fm, l27_fm);
	conv_layer_k1<X28,Y28,Z28,NF28,SFI28,SFW28,SFO28, weights_l28,8,false> (l27_fm, l28_fm);

	add_shortcut<X28,Y28,NF28,Z26,SFO28,SFO26,SFBLK9>(l28_fm, shortcut_9_fm, sum_9);

	relu<X28,Y28,NF28>(sum_9, final_relu);

	average_pool<X28,Y28,XDS3,YDS3, NF28,KDS3, SFBLK9, SFDS3>(final_relu, final_ds);

	fully_connected<NF28,NCLASSES, weights_fc, bias_fc, SFDS3,SFW_FC,SFB_FC,SFO_FC>(final_ds, output_fm);




	write_ofm(output_fm, strm_out);
}


void read_ifm(hls::stream<strmi_t> &strm_in, act_reshp in_feature_map[INPUT1_MEM_SIZE/RESHP_FACTOR], act_reshp shortcut_ifm[INPUT1_MEM_SIZE/RESHP_FACTOR]){
	strmi_t tmpin;
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

void write_ofm(quant_accum ofm[NCLASSES], hls::stream<strmo_t> &strm_out) {
	strmo_t tmpout;
	//Write output fm to stream
	for(int i = 0; i < NCLASSES; i++){
		if(i == (NCLASSES) - 1) tmpout.last = 1;
		else tmpout.last = 0;
		tmpout.data = ofm[i];
		tmpout.keep = 0xF;
		tmpout.strb = 0xF;
		strm_out.write(tmpout);


//#pragma HLS PIPELINE II=8
//		tmpout.data = (quant_act)ofm[i].range(3,0);
////		printf("%d: %d   \n", (int)(quant_act)ofm[i].range(3,0), (int)tmpout.data);
//		tmpout.keep = 0xF;
//		tmpout.strb = 0xF;
//		tmpout.last = 0;
//		strm_out.write(tmpout);
//		tmpout.data = (quant_act)ofm[i].range(7,4);
//		tmpout.keep = 0xF;
//		tmpout.strb = 0xF;
//		tmpout.last = 0;
//		strm_out.write(tmpout);
//		tmpout.data = (quant_act)ofm[i].range(11,8);
//		tmpout.keep = 0xF;
//		tmpout.strb = 0xF;
//		tmpout.last = 0;
//		strm_out.write(tmpout);
//		tmpout.data = (quant_act)ofm[i].range(15,12);
//		tmpout.keep = 0xF;
//		tmpout.strb = 0xF;
//		tmpout.last = 0;
//		strm_out.write(tmpout);
//		tmpout.data = (quant_act)ofm[i].range(19,16);
//		tmpout.keep = 0xF;
//		tmpout.strb = 0xF;
//		tmpout.last = 0;
//		strm_out.write(tmpout);
//		tmpout.data = (quant_act)ofm[i].range(23,20);
//		tmpout.keep = 0xF;
//		tmpout.strb = 0xF;
//		tmpout.last = 0;
//		strm_out.write(tmpout);
//		tmpout.data = (quant_act)ofm[i].range(27,24);
//		tmpout.keep = 0xF;
//		tmpout.strb = 0xF;
//		tmpout.last = 0;
//		strm_out.write(tmpout);
//		if(i == (NCLASSES/RESHP_FACTOR) - 1) tmpout.last = 1;
//		else tmpout.last = 0;
//		tmpout.data = (quant_act)ofm[i].range(31,28);
//		tmpout.keep = 0xF;
//		tmpout.strb = 0xF;
//		strm_out.write(tmpout);
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
							avg = avg >> (sf_i-sf_o);
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
#pragma HLS PIPELINE II=2
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

				sum = sum >> (scale_factor-sf_o);
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

template<params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t sf_i, params_t sf_weights, params_t sf_o, wght_reshp *weights, params_t PE, bool relu>
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
							if(relu) acc = (ap_int<1>)acc[20] == 0 ? acc : (quant_accum)0;
//							tmp_out = (quant_act)acc;
							acc = acc>>((sf_weights+sf_i)-sf_o);
							tmp_out = acc.range(3,0);
//							printf("%X\n\n\n",tmp_out);
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
template<params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t sf_i, params_t sf_weights, params_t sf_o, wght_reshp *weights, params_t PE, bool relu>
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
								if(relu) acc = (ap_int<1>)acc[20] == 0 ? acc : (quant_accum)0;
//								tmp_out = (quant_act)acc;
								acc = acc>>((sf_weights+sf_i)-sf_o);
								tmp_out = acc.range(3,0);
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
template<params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t sf_i, params_t sf_weights, params_t sf_o, wght_reshp *weights, params_t PE, bool relu>
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
							if(relu) acc = (ap_int<1>)acc[20] == 0 ? acc : (quant_accum)0;
//							tmp_out = (quant_act)acc;
//							printf("%X -> ", acc);
							acc = acc>>((sf_weights+sf_i)-sf_o);
							tmp_out = acc.range(3,0);
//							printf("%X -> ", acc);
//							printf("%X\n\n",tmp_out);
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
template<params_t fm_width, params_t fm_height, params_t nbands, params_t nfilters, params_t output_dim, params_t sf_i, params_t sf_weights, params_t sf_o,wght_reshp *weights, params_t PE, bool relu>
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
							acc_even += acc_odd;
							if(relu) acc_even = (ap_int<1>)acc_even[20] == 0 ? acc_even : (quant_accum)0;
//							tmp_out = (quant_act)acc_even;
							acc_even = acc_even>>((sf_weights+sf_i)-sf_o);
							tmp_out = acc_even.range(3,0);
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
void fully_connected(act_reshp input_fm[input_size/RESHP_FACTOR], quant_accum output_fm[nfilters]) {
	quant_accum acc = 0;
	act_reshp tmp_in = 0, tmp_acc = 0;
	wght_reshp tmp_weight = 0;
	int kernel_idx = 0, output_idx = 0, sf = sf_i;

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
				output_fm[i] = acc + bias[i];
				acc = 0;
			}
		}
	}
}

template<params_t width, params_t height, params_t nbands>
void gen_shortcut(act_reshp fm[width*height*nbands/RESHP_FACTOR], act_reshp fm1[width*height*nbands/RESHP_FACTOR], act_reshp fm2[width*height*nbands/RESHP_FACTOR]) {
	for(int i = 0; i < width*height*nbands/RESHP_FACTOR; i++) {
		fm1[i] = fm[i];
		fm2[i] = fm[i];
	}
}

template<params_t width, params_t height, params_t nbands>
void relu(act_reshp in[width*height*nbands/RESHP_FACTOR], act_reshp out[width*height*nbands/RESHP_FACTOR]) {
	quant_act tmp = 0;
	act_reshp tmp_buff_in = 0, tmp_buff_out = 0;
	for(int i = 0; i < width*height*nbands/RESHP_FACTOR; i++) {
		tmp_buff_in = in[i];
		tmp = (quant_act)tmp_buff_in.range(3,0);
		tmp_buff_out.range(3,0) = (ap_int<1>)tmp[3] == 0 ? tmp : (quant_act) 0;
		tmp = (quant_act)tmp_buff_in.range(7,4);
		tmp_buff_out.range(7,4) = (ap_int<1>)tmp[3] == 0 ? tmp : (quant_act) 0;
		tmp = (quant_act)tmp_buff_in.range(11,8);
		tmp_buff_out.range(11,8) = (ap_int<1>)tmp[3] == 0 ? tmp : (quant_act) 0;
		tmp = (quant_act)tmp_buff_in.range(15,12);
		tmp_buff_out.range(15,12) = (ap_int<1>)tmp[3] == 0 ? tmp : (quant_act) 0;
		tmp = (quant_act)tmp_buff_in.range(19,16);
		tmp_buff_out.range(19,16) = (ap_int<1>)tmp[3] == 0 ? tmp : (quant_act) 0;
		tmp = (quant_act)tmp_buff_in.range(23,20);
		tmp_buff_out.range(23,20) = (ap_int<1>)tmp[3] == 0 ? tmp : (quant_act) 0;
		tmp = (quant_act)tmp_buff_in.range(27,24);
		tmp_buff_out.range(27,24) = (ap_int<1>)tmp[3] == 0 ? tmp : (quant_act) 0;
		tmp = (quant_act)tmp_buff_in.range(31,28);
		tmp_buff_out.range(31,28) = (ap_int<1>)tmp[3] == 0 ? tmp : (quant_act) 0;
		out[i] = tmp_buff_out;
	}
}
