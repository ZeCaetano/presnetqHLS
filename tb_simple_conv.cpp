#include "simple_conv.h"
#include <stdio.h>
#include <stdlib.h>

static quant_t image_in[INPUT1_MEM_SIZE];
static quant_t kernel[WEIGHTS_MEM_SIZE];

quant_t hw_image_out[OUT2_FM_MEM_SIZE];
quant_t sw_image_out_1[OUT1_FM_MEM_SIZE];
quant_t sw_image_out_2[OUT2_FM_MEM_SIZE];

//Performs software-only matrix convolution.
void sw_convolution_3D(quant_t *image_in, const quant_t *weights, quant_t *image_out, int nbands, int fm_size, int kernel_size) {
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
					}
				}
			}
//			printf("\n");
//			printf("%X  ", accum);
//			printf("%X\n", (quant_t)accum);
			image_out[ i * fm_size + j] = (quant_t)accum;
		}
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

//	printf("Input Image\n\r");
	for(int k = 0; k < Z1; k++) {
		for (int i = 0; i < X1; i++) {
			for (int j = 0; j < Y1; j++) {
				image_in[(k*X1*Y1)+(i*Y1+j)] = (i+j+k + 1) * 0.01 + ((j + 1)*(k+1))*0.01;
				npixels++;
//				printf("i%d %f ", npixels-1, (float)image_in[(k*X1*Y1)+(i * Y1 + j)]);
			}
//			printf("\n\r");
		}
	}
//	printf("Weights\n\r");
	for(int k = 0; k < Z1; k++) {
		for(int i = 0; i < NF1; i++) {
			for(int j = 0; j < K1; j++) {
				for(int l = 0; l < K1; l++) {
					kernel[(k*NF1*K1*K1)+(i * K1*K1 + (j * K1) + l)] = (i + 1) * 0.1 + (j + 1)*(k+1)*(l+1)*0.1;
//					printf("%f ", (float)kernel[(k*NF1*K1*K1)+(i * K1*K1 + (j * K1) + l)]);
				}
			}
		}
//		printf("\n\r");
	}
	for(int k = 0; k < Z2; k++) {
		for(int i = 0; i < NF2; i++) {
			for(int j = 0; j < K2; j++) {
				for(int l = 0; l < K2; l++) {
					kernel[LAYER1_WEIGHTS + (k*NF2*K2*K2)+(i * K2*K2 + (j * K2) + l)] = (i + 1) * 0.2 + (j + 1)*(k+1)*(l+1)*0.1;
//					printf("%f ", (float)kernel[LAYER1_WEIGHTS + ((k*NF2*K2*K2)+(i * K2*K2 + (j * K2) + l))]);
				}
			}
		}
//		printf("\n\r");
	}

}


int main() {

    hls::stream<strmio_t> sin,so;
    strmio_t vin;
    strmio_t vout;

    init_fm();
//    init_weights_from_file();
//    for(int i = 0; i < WEIGHTS_MEM_SIZE; i++) {
//    	printf("%d\n", (int)kernel[i]);
//    }
//    return 1;

	//--------------------------------------------------------------------------------------------------//
	//-----------------------------------------HARDWARE CONV--------------------------------------------//
	//--------------------------------------------------------------------------------------------------//

////    printf("Sending weights\n");
//    for (int t=0 ; t<WEIGHTS_MEM_SIZE; t++) {
//		vin.data = kernel[t];
//		if(t == WEIGHTS_MEM_SIZE - 1) vin.last = (ap_int<1>)1;
//		else vin.last = (ap_int<1>)0;
//		sin.write(vin);
//    }
    int nweights = 0;
    for (int i = 0 ; i < Z1; i++) {
    	for(int j = 0; j < NF1; j++){
    		for(int x = 0; x < K1*K1; x++){
    			nweights++;
//    			printf("Sending - %d %d %d %f\n", j, i, x, (float)kernel[(i*K1*K1) + (j*Z1*K1*K1) + x]);
				vin.data = kernel[(i*K1*K1) + (j*Z1*K1*K1) + x];
				vin.last = (ap_int<1>)0;
				sin.write(vin);
//				printf("weight sent: b %d -- f %d -- %f\n",i,j,(float)vin.data);
    		}
    	}
	}
    for (int i = 0 ; i < Z2; i++) {
		for(int j = 0; j < NF2; j++){
			for(int x = 0; x < K2*K2; x++){
				nweights++;
//    			printf("Sending - %d %d %d %f\n", j, i, x, (float)kernel[LAYER1_WEIGHTS + (i*K1*K1) + (j*Z1*K1*K1) + x]);
				vin.data = kernel[LAYER1_WEIGHTS + (j*Z2*K2*K2) + (i*K2*K2) + x];
				if(nweights == WEIGHTS_MEM_SIZE) vin.last = (ap_int<1>)1;
				else vin.last = (ap_int<1>)0;
				sin.write(vin);
//				printf("weight sent: b %d -- f %d -- %f\n",i,j,(float)vin.data);
			}
		}
	}
//    printf("Sending fm\n");
#ifdef ARRAYS
	for (int t=0 ; t<INPUT1_MEM_SIZE; t++) {
		vin.data = image_in[t];
		if(t == INPUT1_MEM_SIZE - 1) vin.last = (ap_int<1>)1;
		else vin.last = (ap_int<1>)0;
		sin.write(vin);
//		printf("pixel sent: %f count: %d last: %d\n",image_in[t], t, vin.last);
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
//	exit(0);

	//Hardware computation
    simple_conv(sin, so);

#ifdef ARRAYS
    //Read image_out
	for(int t=0 ; t < OUT2_FM_MEM_SIZE ; t++){
		vout = so.read();
		hw_image_out[t] = vout.data;
//		printf("%f ", vout.data);
		if (vout.last == 1) break;
	}
#else
	for(int i = 0; i < X2*Y2; i++) {
		for(int j = 0; j < Z2; j++) {
			vout = so.read();
			hw_image_out[j*X2*Y2+ i] = vout.data;
			if(vout.last == 1) break;
		}
	}
#endif

	//--------------------------------------------------------------------------------------------------//
	//-----------------------------------------SOFTWARE CONV--------------------------------------------//
	//--------------------------------------------------------------------------------------------------//
	//Layer 1
	for(int i = 0; i < NF1; i++){
		/* Address where the convolutional weights are stored */
		quant_t *fp_weights =
				 kernel +                                       /* start address of params */
				i * (Z1 * K1 * K1); /* kernel of OFM(i) */

		quant_t *image_out_1 =
				sw_image_out_1 +                                        /* base address */
				i * (X1 * Y1);               /* offset (number of images) */

	    sw_convolution_3D(image_in, fp_weights, image_out_1, Z1, X1, K1);
	}
	//Layer 2
//	printf("---------------------SW Layer 2------------------------\n");
	for(int i = 0; i < NF2; i++){
		/* Address where the convolutional weights are stored */
		quant_t *fp_weights =
				 kernel +                                       /* start address of params */
				LAYER1_WEIGHTS + (i * (Z2 * K2 * K2)); /* kernel of OFM(i) */

		quant_t *image_out_2 =
				sw_image_out_2 +                                        /* base address */
				i * (X2 * Y2);               /* offset (number of images) */

	    sw_convolution_3D(sw_image_out_1, fp_weights, image_out_2, Z2, X2, K2);
	}

//    printf("SOFTWARE Output Image 1\n\r");
//    for(int k = 0; k < NF1; k++) {
//		for (int i = 0; i < X1; i++) {
//			for (int j = 0; j < Y1; j++) {
//				printf("%f", (float)sw_image_out_2[(k*X1*Y1) + (i*Y1) + j]);
//			}
//			printf("\n\r");
//		}
//    }


	//--------------------------------------------------------------------------------------------------//
	//-----------------------------------------Results Verification-------------------------------------//
	//--------------------------------------------------------------------------------------------------//
    int err_cnt = 0;
    for(int k = 0; k < NF2; k++) {
		for (int i = 0; i < X2; i++)
			for (int j = 0; j < Y2; j++)
				if (hw_image_out[(k*X2*Y2) + (i*Y2) + j] != sw_image_out_2[(k*X2*Y2) + (i*Y2) + j]) {
					err_cnt++;
					printf("%d - %d,%d: %f != %f\n\r",
						   k, i, j, (float)hw_image_out[(k*X2*Y2) + (i*Y2) + j], (float)sw_image_out_2[(k*X2*Y2) + (i*Y2) + j]);
				}
    }
    printf("\n%d different values\n", err_cnt);

    return err_cnt;
}
