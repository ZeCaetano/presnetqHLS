#include "simple_conv.h"
#include <stdio.h>
#include <stdlib.h>

static quant_t image_in[INPUT1_MEM_SIZE];
static quant_t kernel[WEIGHTS_MEM_SIZE];

float hw_image_out[OUT2_FM_MEM_SIZE];
float sw_image_out_1[OUT1_FM_MEM_SIZE];
float sw_image_out_2[OUT2_FM_MEM_SIZE];

//Performs software-only matrix convolution.
void sw_convolution_3D(quant_t *image_in, const quant_t *weights, float*image_out, int nbands, int fm_size, int kernel_size) {
	for (int i = 0; i < fm_size; i++) {
		for (int j = 0; j < fm_size; j++) {
			float accum = 0;
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

//						printf("SW weight %f - %d\n", weights[weight_1d_idx], weight_1d_idx);
//						printf("SW pixel %f - %d\n", image_in[image_1d_idx], image_1d_idx);

						accum += weights[weight_1d_idx] * ((float) image_in[image_1d_idx] / 255 - 0.5F) / 0.5F;
					}
				}
			}
//			printf("\n");
			image_out[ i * fm_size + j] = accum;
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
				image_in[(k*X1*Y1)+(i*Y1+j)] = (i + 1) * 10 + (j + 1)*(k+1);
				npixels++;
//				printf("i%d %f ", npixels-1, image_in[(k*FM_HEIGHT*FM_WIDTH)+(i * FM_WIDTH + j)]);
			}
//			printf("\n\r");
		}
	}
//	printf("Weights\n\r");
	for(int k = 0; k < Z1; k++) {
		for(int i = 0; i < NF1; i++) {
			for(int j = 0; j < K1; j++) {
				for(int l = 0; l < K1; l++) {
					kernel[(k*NF1*K1*K1)+(i * K1*K1 + (j * K1) + l)] = (i + 1) * 0.01 + (j + 1)*(k+1)*(l+1)*0.1;
//					printf("%f ", (float)kernel[(k*N_FILTERS*KERNEL_SIZE*KERNEL_SIZE)+(i * KERNEL_SIZE*KERNEL_SIZE + (j * KERNEL_SIZE) + l)]);
				}
			}
		}
	}
	for(int k = 0; k < Z2; k++) {
			for(int i = 0; i < NF2; i++) {
				for(int j = 0; j < K2; j++) {
					for(int l = 0; l < K2; l++) {
						kernel[LAYER1_WEIGHTS + (k*NF2*K2*K2)+(i * K2*K2 + (j * K2) + l)] = (i + 1) * 0.02 + (j + 1)*(k+1)*(l+1)*0.1;
	//					printf("%f ", (float)kernel[(k*N_FILTERS*KERNEL_SIZE*KERNEL_SIZE)+(i * KERNEL_SIZE*KERNEL_SIZE + (j * KERNEL_SIZE) + l)]);
					}
				}
			}
		}
//    	printf("\n\r");

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

//    printf("Sending weights\n");
    for (int t=0 ; t<WEIGHTS_MEM_SIZE; t++) {
		vin.data = kernel[t];
		if(t == WEIGHTS_MEM_SIZE - 1) vin.last = (ap_int<1>)1;
		else vin.last = (ap_int<1>)0;
		sin.write(vin);
//		printf("weight sent: %f\n",vin.data);
	}
//    printf("Sending fm\n");
	for (int t=0 ; t<INPUT1_MEM_SIZE; t++) {
		vin.data = image_in[t];
		if(t == INPUT1_MEM_SIZE - 1) vin.last = (ap_int<1>)1;
		else vin.last = (ap_int<1>)0;
		sin.write(vin);
//		printf("pixel sent: %f count: %d last: %d\n",image_in[t], t, vin.last);
	}
	printf("finish sending weights and fm\n");
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
	//		if(layer_id == 1) printf("%f-%d\n", tmpin.data, i);
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

		float *image_out_1 =
				(float *) sw_image_out_1 +                                        /* base address */
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

		float *image_out_2 =
				(float *) sw_image_out_2 +                                        /* base address */
				i * (X2 * Y2);               /* offset (number of images) */


	    sw_convolution_3D(sw_image_out_1, fp_weights, image_out_2, Z2, X2, K2);
	}

//    printf("SOFTWARE Output Image 1\n\r");
//    for(int k = 0; k < NF1; k++) {
//		for (int i = 0; i < X1; i++) {
//			for (int j = 0; j < Y1; j++) {
//				printf("%f-%d\n", sw_image_out_1[(k*X1*Y1) + (i*Y1) + j], (k*X1*Y1) + (i*Y1) + j);
//			}
////			printf("\n\r");
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
						   k, i, j, hw_image_out[(k*X2*Y2) + (i*Y2) + j], sw_image_out_2[(k*X2*Y2) + (i*Y2) + j]);
				}
    }
    printf("\n%d different values\n", err_cnt);

    return err_cnt;
}
