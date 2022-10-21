#include "simple_conv.h"


static quant_t image_in[FM_HEIGHT * FM_WIDTH * N_BANDS];
static quant_t kernel[KERNEL_SIZE * KERNEL_SIZE*N_BANDS*N_FILTERS];

float hw_image_out[FM_HEIGHT * FM_WIDTH * N_FILTERS];
float sw_image_out[FM_HEIGHT * FM_WIDTH * N_FILTERS];

//Performs software-only matrix convolution.
void sw_convolution_3D(quant_t *image_in, const quant_t *weights, float*image_out) {
	for (int i = 0; i < FM_HEIGHT; i++) {
		for (int j = 0; j < FM_WIDTH; j++) {
			float accum = 0;
			for (int k = 0; k < N_BANDS; k++) {
				for (int x = 0; x < KERNEL_SIZE; x++) {
					for (int y = 0; y < KERNEL_SIZE; y++) {
						/* Weights matrix index */
						int weight_1d_idx =
								k * (KERNEL_SIZE * KERNEL_SIZE) + /* channels */
								x * KERNEL_SIZE + y;                   /* planar coordinate */
//						printf("%d ", weight_1d_idx);

						/* Input image index */
						int image_1d_idx =
								k * (FM_HEIGHT * FM_WIDTH) +          /* OFM */
								(x + i) * FM_HEIGHT + (y + j);           /* pixel */

						accum += weights[weight_1d_idx] * ((float) image_in[image_1d_idx] / 255 - 0.5F) / 0.5F;
					}
				}
			}
//			printf("\n");
			image_out[ i * FM_WIDTH + j] = accum;
		}
	}

}

int main() {
//    printf("Input Image\n\r");
    for(int k = 0; k < N_BANDS; k++) {
		for (int i = 0; i < FM_HEIGHT; i++) {
			for (int j = 0; j < FM_WIDTH; j++) {
				image_in[(k*FM_HEIGHT*FM_WIDTH)+(i * FM_WIDTH + j)] = (i + 1) * 10 + (j + 1)*(k+1);
//				printf("%f ", (float)image_in[(k*FM_HEIGHT*FM_WIDTH)+(i * FM_WIDTH + j)]);
			}
//			printf("\n\r");
		}
    }

//    printf("Weights\n\r");
    for(int k = 0; k < N_BANDS; k++) {
    	for(int i = 0; i < N_FILTERS; i++) {
    		for(int j = 0; j < KERNEL_SIZE; j++) {
    			for(int l = 0; l < KERNEL_SIZE; l++) {
    				kernel[(k*N_FILTERS*KERNEL_SIZE*KERNEL_SIZE)+(i * KERNEL_SIZE*KERNEL_SIZE + (j * KERNEL_SIZE) + l)] = (i + 1) * 0.01 + (j + 1)*(k+1)*(l+1)*0.1;
//					printf("%f ", (float)kernel[(k*N_FILTERS*KERNEL_SIZE*KERNEL_SIZE)+(i * KERNEL_SIZE*KERNEL_SIZE + (j * KERNEL_SIZE) + l)]);
    			}
    		}
    	}
//    	printf("\n\r");
    }

    hls::stream<strmio_t> sin,so;

    strmio_t vin;
    strmio_t vout;

	for (int t=0 ; t<WEIGHTS_MEM_SIZE; t++) {
		vin.data = kernel[t];
		if(t == WEIGHTS_MEM_SIZE - 1) vin.last = (ap_int<1>)1;
		else vin.last = (ap_int<1>)0;
		sin.write(vin);
//		printf("weight sent: %f\n",vin.data);
	}
	for (int t=0 ; t<FM_MEM_SIZE/4; t++) {
		vin.data = image_in[t];
		if(t == FM_MEM_SIZE - 1) vin.last = (ap_int<1>)1;
		else vin.last = (ap_int<1>)0;
		sin.write(vin);
//		printf("pixel sent: %f\n",vin.data);
	}

	//Hardware computation
    simple_conv(sin, so);

    //Read image_out
	for(int t=0 ; t < FM_WIDTH * FM_HEIGHT * N_FILTERS ; t++){
		vout = so.read();
		hw_image_out[t] = vout.data;
		printf("%f ", vout.data);
		if (vout.last == 1) break ;
	}

	for(int i = 0; i < N_FILTERS; i++){
		/* Address where the convolutional weights are stored */
		quant_t *fp_weights =
				 kernel +                                       /* start address of params */
				i * (N_BANDS * KERNEL_SIZE * KERNEL_SIZE); /* kernel of OFM(i) */

		float *image_out =
				(float *) sw_image_out +                                        /* base address */
				i * (FM_HEIGHT * FM_WIDTH);               /* offset (number of images) */


	    sw_convolution_3D(image_in, fp_weights, image_out);
	}

//    printf("Output Image\n\r");
//    for(int k = 0; k < N_FILTERS; k++) {
//		for (int i = 0; i < FM_HEIGHT; i++) {
//			for (int j = 0; j < FM_WIDTH; j++) {
//				printf("%f ", sw_image_out[(k+1)*i * FM_WIDTH + j]);
//			}
//			printf("\n\r");
//		}
//    }

    int err_cnt = 0;
    for(int k = 0; k < N_FILTERS; k++) {
		for (int i = 0; i < FM_HEIGHT; i++)
			for (int j = 0; j < FM_WIDTH; j++)
				if (hw_image_out[(k+1)*i * FM_WIDTH + j] != sw_image_out[(k+1)*i * FM_WIDTH + j]) {
					err_cnt++;
					printf("%d,%d: %f != %f\n\r",
						   i, j, hw_image_out[(k+1)*i * FM_WIDTH + j], sw_image_out[(k+1)*i * FM_WIDTH + j]);
				}
    }

    return err_cnt;
}
