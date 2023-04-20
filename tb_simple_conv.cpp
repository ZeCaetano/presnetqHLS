#include "simple_conv.h"
#include <stdio.h>
#include <stdlib.h>

static quant_act image_in[INPUT1_MEM_SIZE];
//static quant_wght kernel[LAYER1_WEIGHTS+LAYER3_WEIGHTS+LAYER2_WEIGHTS];
//static quant_bias bias[NCLASSES];

//quant_act hw_image_out[OUTPUT_MEM_SIZE];
quant_accum hw_image_out[NCLASSES];
quant_act sw_image_out_1[OUT1_MEM_SIZE];
quant_act sw_image_out_2[OUT2_MEM_SIZE];
quant_act sw_image_out_3[OUT3_MEM_SIZE];
quant_act sw_image_out_4[OUT3_MEM_SIZE];
quant_act sw_image_out_5[OUT3_MEM_SIZE];
quant_act sw_image_out[NCLASSES];
quant_act sw_image_out_ds[OUT3_MEM_SIZE];

//Performs software-only matrix convolution.
void sw_convolution_3D(quant_act *image_in, const quant_wght *weights, quant_act *image_out, int nbands, int fm_size, int kernel_size, bool relu) {
	for (int i = 0; i < fm_size; i++) {
		for (int j = 0; j < fm_size; j++) {
			quant_accum accum = 0;
			for (int k = 0; k < nbands; k++) {
				for (int x = 0; x < kernel_size; x++) {
					for (int y = 0; y < kernel_size; y++) {
						/* Weights matrix index */
						int weight_1d_idx =
								k * (kernel_size * kernel_size) + /* channels */
								x * kernel_size + y;                   /* planar coordinate */
						/* Input image index */
						int image_1d_idx =
								k * (fm_size * fm_size) +          /* OFM */
								(x + i) * fm_size + (y + j);           /* pixel */

//						printf("TB - %d %d %f\n", k, (x*kernel_size) + y, (float)weights[weight_1d_idx]);
//						printf("SW pixel %f - %d\n", image_in[image_1d_idx], image_1d_idx);

						accum += weights[weight_1d_idx] * image_in[image_1d_idx];
//						printf("%X\n", accum);
					}
				}
			}
//			printf("\n");
//			printf("%X  ", accum);
//			printf("%X\n", (quant_act)accum);
			image_out[ i * fm_size + j] = (quant_act)accum;
			if(relu)
				image_out[ i * fm_size + j] = (int)image_out[ i * fm_size + j] > 0 ? image_out[ i * fm_size + j] : (quant_act)0;
		}
	}

}

void sw_convolution_3D_k2(quant_act *image_in, const quant_wght *weights, quant_act *image_out, int nbands, int fm_size, int kernel_size, int output_size, bool relu) {
	//stride = 2
	for (int i = 0; i < fm_size-1; i+=2) {
		for (int j = 0; j < fm_size-1; j+=2) {
			int accum = 0;
			for (int k = 0; k < nbands; k++) {
				for (int x = 0; x < kernel_size; x++) {
					for (int y = 0; y < kernel_size; y++) {
						/* Weights matrix index */
						int weight_1d_idx =
								k * (kernel_size * kernel_size) + /* channels */
								x * kernel_size + y;                   /* planar coordinate */
						/* Input image index */
						int image_1d_idx =
								k * (fm_size * fm_size) +          /* OFM */
								(x + i) * fm_size + (y + j);           /* pixel */

//						printf("TB - %d %d %f\n", k, (x*kernel_size) + y, (float)weights[weight_1d_idx]);
//						printf("SW pixel %f - %d\n", image_in[image_1d_idx], image_1d_idx);

						accum += weights[weight_1d_idx] * image_in[image_1d_idx];
//						printf("%X\n", accum);
					}
				}
			}
//			printf("\n");
//			printf("%X  ", accum);
//			printf("%X\n", (quant_act)accum);
//			printf("index %d\n", (i/2) * output_size + (j/2));
			image_out[ (i/2) * output_size + (j/2)] = (quant_act)accum;
			if(relu)
				image_out[ (i/2) * output_size + (j/2)] = (int)image_out[ (i/2) * output_size + (j/2)] > 0 ? image_out[ (i/2) * output_size + (j/2)] : (quant_act)0;
		}
	}

}


void average_pooling(quant_act *image_in, quant_act *image_out, int fm_size, int out_size, int nbands, int kernel_size){
	int accum;
	int avg;
	for(int z = 0; z < nbands; z++){
		for (int i = 0; i < fm_size-1; i+=2) {
			for (int j = 0; j < fm_size-1; j+=2) {
				accum = 0;
				for(int k = 0; k < kernel_size; k++){
					for(int l = 0; l < kernel_size; l++){
						accum += image_in[(z*fm_size*fm_size) + (i+k)*fm_size + j+l];
					}
				}
				avg = accum / (kernel_size*kernel_size);
//				printf("SW idx: %d accum: %d    avg: %d\n", z*out_size*out_size + (i/2)*out_size + (j/2), accum, avg);

				image_out[z*out_size*out_size + (i/2)*out_size + (j/2)] = (quant_act) avg;
			}
		}
	}

}

void sum_shorctut(quant_act *conv_fm, quant_act *shortcut, quant_act *fm_out, int fm_size, int nbands_conv, int nbands_shortcut){
	int sum = 0;
	for(int z = 0; z < nbands_conv; z++){
		for (int i = 0; i < fm_size; i++) {
			for (int j = 0; j < fm_size; j++) {
				if(z >= nbands_shortcut) {
					sum = conv_fm[z*fm_size*fm_size + i*fm_size + j];
				}
				else {
					sum = conv_fm[z*fm_size*fm_size + i*fm_size + j] + shortcut[z*fm_size*fm_size + i*fm_size + j];
				}
				fm_out[z*fm_size*fm_size + i*fm_size + j] = (quant_act) sum;
			}
		}
	}
}

void fully_connected(quant_act *input_fm, quant_accum *output_fm, quant_wght *weights, quant_bias *bias, int input_size, int nfilters) {
	int accum = 0, tmp = 0;
	for(int i = 0; i < nfilters; i++) {
		for(int j = 0; j < input_size; j++) {
			accum += input_fm[j]*weights[i*input_size + j];
		}
		tmp = accum + bias[i];
		output_fm[i] = tmp;
		accum = 0;
	}
}


//void init_weights_from_file(){
//	FILE *fweights;
//	char tmp[(3440)/2];
//
//	// Open weights file
//	if ((fweights = fopen("simple_weights.bin", "rb")) == NULL) {
//		fprintf(stderr, "unable to open file <simple_weights.bin>\n");
//		exit(1);
//	}
//	// read weights
////	fread((char *)kernel, 1,(3440)/2, fweights);

//	fread(tmp, 1,(3440)/2, fweights);
//	for(int i = 0; i < WEIGHTS_MEM_SIZE; i+=2) {
//		kernel[i] = tmp[i] >> 4;
//		kernel[i+1] = tmp[i] & 0b00001111;
//	}
//
//
//}

void init_fm(){

	int npixels = 0;
//	quant_act values1[16] = {200,142,222,133,247,35,96,72,11,231,194,166,55,182,175,89};
	quant_act values1[16] = {-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7};
	quant_wght values2[4] = {-2,-1,1};
	quant_wght values3[4] = {1,-1,-2};
	quant_wght values4[4] = {-1,1,-2};
	quant_act pixel = 0;
	char buff_image[INPUT1_MEM_SIZE];

	FILE *input_file = fopen("input_image.bin", "rb");
	fread(buff_image, sizeof(char), INPUT1_MEM_SIZE, input_file);


//	for(int i = 0; i < INPUT1_MEM_SIZE; i++) {
//		pixel = (ap_int<4>)buff_image[i];
//		pixel = v.range(3,0);
//		printf("%d ", (int)v);
//	}

	//	printf("Input Image\n\r");
		for(int k = 0; k < Z1; k++) {
			for (int i = 0; i < X1; i++) {
				for (int j = 0; j < Y1; j++) {
					pixel = (ap_int<4>)buff_image[(k*X1*Y1)+(i*Y1)+j];
					pixel = pixel.range(3,0);
					image_in[(k*X1*Y1)+(i*Y1)+j] = pixel;
					npixels++;
//					printf("%d ", (int)image_in[(k*X1*Y1)+(i * Y1 + j)]);
				}
//				printf("\n\r");
			}
		}
}


int main() {

    hls::stream<strmi_t> sin;
    hls::stream<strmo_t> so;
    strmi_t vin;
    strmo_t vout;
    printf("Start\n");

    init_fm();
    printf("FM initialized\n");

//    init_weights_from_file();
//    for(int i = 0; i < WEIGHTS_MEM_SIZE; i++) {
//    	printf("%d\n", (int)kernel[i]);
//    }
//    return 1;

	//--------------------------------------------------------------------------------------------------//
	//-----------------------------------------HARDWARE CONV--------------------------------------------//
	//--------------------------------------------------------------------------------------------------//

//    printf("Sending weights\n");
//    for (int t=0 ; t<WEIGHTS_MEM_SIZE; t++) {
//		vin.data = kernel[t];
//		if(t == WEIGHTS_MEM_SIZE - 1) vin.last = (ap_int<1>)1;
//		else vin.last = (ap_int<1>)0;
//		sin.write(vin);
//    }
//    int nweights = 0;
//    for (int i = 0 ; i < Z1; i++) {
//    	for(int j = 0; j < NF1; j++){
//    		for(int x = 0; x < K1*K1; x++){
//    			nweights++;
////    			printf("Sending - %d %d %d %f\n", j, i, x, (float)kernel[(i*K1*K1) + (j*Z1*K1*K1) + x]);
//				vin.data = kernel[(i*K1*K1) + (j*Z1*K1*K1) + x];
//				vin.last = (ap_int<1>)0;
//				sin.write(vin);
////				printf("weight sent: b %d -- f %d -- %f\n",i,j,(float)vin.data);
//    		}
//    	}
//	}
//    for (int i = 0 ; i < Z2; i++) {
//		for(int j = 0; j < NF2; j++){
//			for(int x = 0; x < K2*K2; x++){
//				nweights++;
////    			printf("Sending - %d %d %d %f\n", j, i, x, (float)kernel[LAYER1_WEIGHTS + (i*K1*K1) + (j*Z1*K1*K1) + x]);
//				vin.data = kernel[LAYER1_WEIGHTS + (j*Z2*K2*K2) + (i*K2*K2) + x];
//				if(nweights == WEIGHTS_MEM_SIZE) vin.last = (ap_int<1>)1;
//				else vin.last = (ap_int<1>)0;
//				sin.write(vin);
////				printf("weight sent: b %d -- f %d -- %f\n",i,j,(float)vin.data);
//			}
//		}
//	}
    printf("Sending fm\n");
    for(int i = 0; i < NPATCHES; i++){
	#ifdef ARRAYS
		for (int t=0 ; t<X1*Y1; t++) {
			for(int j = 0; j < Z1; j++) {
				vin.data = image_in[(j*X1*Y1) + t];
				if(t == INPUT1_MEM_SIZE - 1) vin.last = (ap_int<1>)1;
				else vin.last = (ap_int<1>)0;
				sin.write(vin);
//				printf("pixel sent: %d \n",(int)image_in[(j*X1*Y1) + t]);
			}
		}
	#else
		for (int t=0 ; t<X1*Y1; t++) {
			for(int j = 0; j < Z1; j++) {
				vin.data = image_in[(j*X1*Y1) + t];
				if(t == INPUT1_MEM_SIZE - 1) vin.last = (ap_int<1>)1;
				else vin.last = (ap_int<1>)0;
				sin.write(vin);
		//		printf("pixel sent: %f count: %d last: %d\n",image_in[t], t, vin.last);
			}
		}
	#endif
    }
//	exit(0);
    printf("Start HW\n");
	//Hardware computation
    for(int i = 0; i < NPATCHES; i++)
    	simple_conv(sin, so);

    for(int i = 0; i < NPATCHES; i++){
//	#ifdef ARRAYS
//// 		Read image_out
//		for(int t=0 ; t < OUT2_MEM_SIZE ; t++){
//			vout = so.read();
//			hw_image_out[t] = vout.data;
////			printf("%d ", (int)vout.data);
//			if (vout.last == 1) break;
//		}
//		for(int l = 0; l < X3*Y3; l++) {
//			for(int j = 0; j < NF3; j++) {
//				vout = so.read();
//				hw_image_out[(j*X3*Y3) + l] = vout.data;
////				printf("idx-%d  %d\n", (j*X3*Y3) + l, (int)vout.data);
//				if(vout.last == 1) break;
//			}
//		}
    	printf("Reading HW\n");
		for(int j = 0; j < NCLASSES; j++) {
			vout = so.read();
			hw_image_out[j] = vout.data;
//				printf("idx-%d  %d\n", (j*X3*Y3) + l, (int)vout.data);
			if(vout.last == 1) break;
		}
//		for(int l = 0; l < XDS*YDS; l++) {
//					for(int j = 0; j < ZDS; j++) {
//						vout = so.read();
//						hw_image_out[(j*XDS*YDS) + l] = vout.data;
//		//				printf("idx-%d  %d\n", (j*X3*Y3) + l, (int)vout.data);
//						if(vout.last == 1) break;
//					}
//				}
    }
//    for(int i = 0; i < OUT2_MEM_SIZE; i++){
//    	sw_image_out_2[i] = 0;
//    }
//
//	//--------------------------------------------------------------------------------------------------//
//	//-----------------------------------------SOFTWARE CONV--------------------------------------------//
//	//--------------------------------------------------------------------------------------------------//
//	//Layer 1
//	for(int i = 0; i < NF1; i++){
//		/* Address where the convolutional weights are stored */
//		quant_wght *fp_weights =
//				 kernel +                                       /* start address of params */
//				i * (Z1 * K1 * K1); /* kernel of OFM(i) */
//
//		quant_act *image_out_1 =
//				sw_image_out_1 +                                        /* base address */
//				i * (X2 * Y2);               /* offset (number of images) */
//
//	    sw_convolution_3D(image_in, fp_weights, image_out_1, Z1, X1, K1, false);
//	}
//	//Layer 2
////	printf("---------------------SW Layer 2------------------------\n");
//	for(int i = 0; i < NF2; i++){
//		/* Address where the convolutional weights are stored */
//		quant_wght *fp_weights =
//				 kernel +                                       /* start address of params */
//				LAYER1_WEIGHTS + (i * (Z2 * K2 * K2)); /* kernel of OFM(i) */
//
//		quant_act *image_out_2 =
//				sw_image_out_2 +                                        /* base address */
//				i * (X3 * Y3);               /* offset (number of images) */
//
////	    sw_convolution_3D(sw_image_out_1, fp_weights, image_out_2, Z2, X2, K2, false);
//		sw_convolution_3D_k2(sw_image_out_1, fp_weights, image_out_2, Z2, X2, K2, X3, false);
//	}
////	printf("---------------------SW Layer 3------------------------\n");
//		for(int i = 0; i < NF3; i++){
//			/* Address where the convolutional weights are stored */
//			quant_wght *fp_weights =
//					 kernel +                                       /* start address of params */
//					LAYER1_WEIGHTS + LAYER2_WEIGHTS + (i * (Z3 * K3 * K3)); /* kernel of OFM(i) */
//
//			quant_act *image_out_3 =
//					sw_image_out_3 +                                        /* base address */
//					i * (X3 * Y3);               /* offset (number of images) */
//
//		    sw_convolution_3D(sw_image_out_2, fp_weights, image_out_3, Z3, X3, K3, false);
//		}
//
//
//	average_pooling(image_in, sw_image_out_ds, X1, X13, Z13, 2);
//
//	sum_shorctut(sw_image_out_3, sw_image_out_ds, sw_image_out_4, X3, NF3, Z1);
//
//	average_pooling(sw_image_out_4, sw_image_out_5, X3, X13, NF13, 2);
//	quant_wght *fp_weights_fc = kernel + LAYER1_WEIGHTS + LAYER2_WEIGHTS + LAYER3_WEIGHTS;
//	fully_connected(sw_image_out_5, sw_image_out, fp_weights_fc, bias, NF3, NCLASSES);

//	printf("Input SW:\n");
//	for(int k = 0; k < Z1; k++) {
//		for (int i = 0; i < X1; i++) {
//			for (int j = 0; j < Y1; j++) {
//				printf("  %d", (int)image_in[(k*X1*Y1) + (i*Y1) + j]);
//			}
//			printf("\n\r");
//		}
//		printf("\n\r");
//	}
//	printf("DS output:\n");
//	for(int k = 0; k < Z1; k++) {
//		for (int i = 0; i < XDS; i++) {
//			for (int j = 0; j < YDS; j++) {
//				printf("  %d", (int)sw_image_out_ds[(k*XDS*YDS) + (i*YDS) + j]);
//			}
//			printf("\n\r");
//		}
//		printf("\n\r");
//	}
//	printf("HARDWARE Output DS Image\n\r");
//	for(int k = 0; k < ZDS; k++) {
//		for (int i = 0; i < XDS; i++) {
//			for (int j = 0; j < YDS; j++) {
//				printf("%d ", (int)hw_image_out[(k*X3*Y3) + (i*Y3) + j]);
//			}
//			printf("\n\r");
//		}
//		printf("%d\n\r", k);
//	}

//	    printf("HARDWARE Output Image\n\r");
//	    for(int k = 0; k < NF3; k++) {
//			for (int i = 0; i < X3; i++) {
//				for (int j = 0; j < Y3; j++) {
//					printf("%d ", (int)hw_image_out[(k*X3*Y3) + (i*Y3) + j]);
//				}
//				printf("\n\r");
//			}
//			printf("%d\n\r", k);
//	    }
//
//    printf("SOFTWARE Output Image 2\n\r");
//    for(int k = 0; k < Z3; k++) {
//		for (int i = 0; i < X3; i++) {
//			for (int j = 0; j < Y3; j++) {
//				printf("%d ", (int)sw_image_out_2[(k*X3*Y3) + (i*Y3) + j]);
//			}
//			printf("\n\r");
//		}
//		printf("%d\n\r", k);
//    }
//    printf("SOFTWARE Output Image\n\r");
//        for(int k = 0; k < NF3; k++) {
//    		for (int i = 0; i < XDS2; i++) {
//    			for (int j = 0; j < YDS2; j++) {
//    				printf("%d ", (int)sw_image_out_5[(k*X3*Y3) + (i*Y3) + j]);
//    			}
//    			printf("\n\r");
//    		}
//    		printf("%d\n\r", k);
//        }

//	printf("Sofware output image after fully connected layer\n");
//	for(int i = 0; i < NCLASSES; i++) {
//		printf("%d ", (int)sw_image_out[i]);
//	}
//	printf("\n");
//
	printf("HARDWARE output image\n");
	for(int i = 0; i < NCLASSES; i++) {
		printf("%d ", (int)hw_image_out[i]);
	}
	printf("\n");

	//--------------------------------------------------------------------------------------------------//
	//-----------------------------------------Results Verification-------------------------------------//
	//--------------------------------------------------------------------------------------------------//
    int err_cnt = 0;
//    for(int k = 0; k < NF3; k++) {
//		for (int i = 0; i < X3; i++)
//			for (int j = 0; j < Y3; j++)
//				if (hw_image_out[(k*X3*Y3) + (i*Y3) + j] != sw_image_out[(k*X3*Y3) + (i*Y3) + j]) {
//					err_cnt++;
//					printf("%d - %d,%d: %d != %d\n\r",
//						   k, i, j, (int)hw_image_out[(k*X3*Y3) + (i*Y3) + j], (int)sw_image_out[(k*X3*Y3) + (i*Y3) + j]);
//				}
//    }
//    for(int k = 0; k < NCLASSES; k++) {
//			if (hw_image_out[k] != sw_image_out[k]) {
//				err_cnt++;
//				printf("%d - %d != %d\n\r",
//					   k, (int)hw_image_out[k], (int)sw_image_out[k]);
//			}
//	}
//    for(int k = 0; k < ZDS; k++) {
//   		for (int i = 0; i < XDS; i++)
//   			for (int j = 0; j < YDS; j++)
//   				if (hw_image_out[(k*XDS*YDS) + (i*YDS) + j] != sw_image_out_ds[(k*XDS*YDS) + (i*YDS) + j]) {
//   					err_cnt++;
//   					printf("%d - %d,%d: %d != %d\n\r",
//   						   k, i, j, (int)hw_image_out[(k*XDS*YDS) + (i*YDS) + j], (int)sw_image_out_ds[(k*XDS*YDS) + (i*YDS) + j]);
//   				}
//       }
//    printf("\n%d different values\n", err_cnt);
    return err_cnt;
}
