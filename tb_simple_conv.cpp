#include "simple_conv.h"
#include <stdio.h>
#include <stdlib.h>

static quant_act image_in[INPUT1_MEM_SIZE*NPATCHES];
static act_reshp reshp_image_in[NPATCHES*INPUT1_MEM_SIZE/RESHP_FACTOR];

data_out hw_image_out[NPATCHES*NCLASSES];
//quant_act hw_image_out[OUT_SIZE];
data_out sw_image_out[NPATCHES*NCLASSES] = {-19.8750,  -0.8750,  -9.5000, -23.8750, -29.2500, -17.3750, -32.1250, -25.2500, -31.8750,  -8.5000,   7.7500, -15.6250, -29.7500, -29.3750, -30.1250, -25.1250, -35.3750, -19.8750,  -4.8750, -19.8750, -34.7500, -26.5000, -51.5000, -44.0000, -25.6250, -13.8750,  12.6250, -11.5000, -32.3750, -39.1250, -35.1250, -25.7500, 1.1250, -14.1250, -19.6250,  -7.7500, -10.6250, -34.0000, -10.1250, 12.3750, -42.2500, -26.7500, -31.1250, -14.6250, -25.5000, -28.8750, -30.7500, -38.2500, -15.1250, -18.1250, -31.1250, -32.7500,  -2.7500,  -8.7500, -20.3750, -22.2500, -36.8750,  -6.5000, -13.3750, -28.1250, -35.2500,  14.6250, -13.0000, -29.0000, -27.1250, -16.6250,   7.3750, -14.2500, -15.8750, -28.6250, -43.6250, -35.7500, -13.5000, -15.6250, -14.3750, -16.7500,  -5.5000, -20.2500, -31.5000, -26.0000,-18.2500, -11.3750, -23.3750, -33.7500, -17.0000, -22.6250, -39.6250, -38.2500, -36.7500,   0.8750,  -6.7500, -18.6250, -30.5000, -25.3750, -24.8750, -26.2500};
//data_out sw_image_out[NPATCHES*NCLASSES] = {-19.8750,  -0.8750,  -9.5000, -23.8750, -29.2500, -17.3750, -32.1250, -25.2500, -31.8750,  -8.5000,   7.7500, -15.6250, -29.7500, -29.3750, -30.1250, -25.1250};
//data_out sw_image_out[NPATCHES*NCLASSES] = {-19.8750,  -0.8750,  -9.5000, -23.8750, -29.2500, -17.3750, -32.1250, -25.2500, -31.8750,  -8.5000,   7.7500, -15.6250, -29.7500, -29.3750, -30.1250, -25.1250, -35.3750, -19.8750,  -4.8750, -19.8750, -34.7500, -26.5000, -51.5000, -44.0000, -25.6250, -13.8750,  12.6250, -11.5000, -32.3750, -39.1250, -35.1250, -25.7500};


void init_fm(){

	int npixels = 0;

	char buff_image[INPUT1_MEM_SIZE];
	unsigned int reshp_buff_image[INPUT1_MEM_SIZE/RESHP_FACTOR];
	quant_act pixel = 0;
	FILE *input_files[6];
	FILE *reshp_files[6];
	input_files[0] = fopen("input_image.bin", "rb");
	input_files[1] = fopen("input_image_1.bin", "rb");
	input_files[2] = fopen("input_image_2.bin", "rb");
	input_files[3] = fopen("input_image_3.bin", "rb");
	input_files[4] = fopen("input_image_4.bin", "rb");
	input_files[5] = fopen("input_image_5.bin", "rb");

	reshp_files[0] = fopen("reshp_input_image.bin", "rb");
	reshp_files[1] = fopen("reshp_input_image_1.bin", "rb");
	reshp_files[2] = fopen("reshp_input_image_2.bin", "rb");
	reshp_files[3] = fopen("reshp_input_image_3.bin", "rb");
	reshp_files[4] = fopen("reshp_input_image_4.bin", "rb");
	reshp_files[5] = fopen("reshp_input_image_5.bin", "rb");
//	FILE *input_file = fopen("input_image_50.bin", "rb");


	//	printf("Input Image\n\r");
//	for(int n = 0; n < NPATCHES; n++) {
//		fread(buff_image, sizeof(char), INPUT1_MEM_SIZE, input_files[n]);
//		for(int k = 0; k < Z1; k++) {
//			for (int i = 0; i < X1; i++) {
//				for (int j = 0; j < Y1; j++) {
//					pixel = (ap_int<4>)buff_image[(k*X1*Y1)+(i*Y1)+j];
//					pixel = pixel.range(3,0);
//					image_in[(n*INPUT1_MEM_SIZE)+((k*X1*Y1)+(i*Y1)+j)] = pixel;
//					npixels++;
////					printf("%d ", (int)image_in[(k*X1*Y1)+(i * Y1 + j)]);
//				}
////				printf("\n\r");
//			}
//		}
//	}
	for(int n = 0; n < NPATCHES; n++){
//		printf("%d\n", n);
		fread(reshp_buff_image, sizeof(int), INPUT1_MEM_SIZE/RESHP_FACTOR, reshp_files[n]);
		for(int i = 0; i < INPUT1_MEM_SIZE/RESHP_FACTOR; i++){
			reshp_image_in[(n*INPUT1_MEM_SIZE/RESHP_FACTOR) + i] = (act_reshp)reshp_buff_image[i];
//			printf("%d ", (unsigned int)reshp_image_in[i]);
		}
	}
	printf("\n");


}


int main() {

    hls::stream<strmi_t> sin[NPATCHES];
    hls::stream<strmo_t> so[NPATCHES];
    strmi_t vin;
    strmo_t vout;
//    printf("Start\n");

    init_fm();
    printf("FM initialized\n");


    for(int i = 0; i < NPATCHES; i++){
		for(int j = 0; j < INPUT1_MEM_SIZE/RESHP_FACTOR; j++){
			printf("%d ", (int) reshp_image_in[i*INPUT1_MEM_SIZE/RESHP_FACTOR + j]);
		}
		printf("\n");

		if(i == NPATCHES-1) vin.data = 1;
		else vin.data = 0;
		sin[i].write(vin);
		for (int t=0 ; t<X1*Y1*Z1/RESHP_FACTOR; t++) {
			vin.data = reshp_image_in[(i*INPUT1_MEM_SIZE/RESHP_FACTOR) + t];
			if(t == INPUT1_MEM_SIZE/RESHP_FACTOR - 1) vin.last = (ap_int<1>)1;
			else vin.last = (ap_int<1>)0;
			sin[i].write(vin);
		//		printf("pixel sent: %d \n",(int)image_in[(j*X1*Y1) + t]);
		}
    }
//	exit(0);
    printf("Start HW\n");
	//Hardware computation
    int last = 0;
    for(int i = 0; i < NPATCHES; i++) {
    	if(i == NPATCHES-1) last = 1;
    	simple_conv(sin[i], so[i]);
    }

    for(int i = 0; i < NPATCHES; i++){

//    	printf("Reading HW\n");
		for(int j = 0; j < NCLASSES; j++) {
			vout = so[i].read();
			hw_image_out[i*NCLASSES + j] = vout.data;
//				printf("idx-%d  %d\n", (j*X3*Y3) + l, (int)vout.data);
			if(vout.last == 1) {
				printf("last!\n");
				break;
			}
		}
    }

    int err_cnt = 0;

    for(int i = 0; i < NPATCHES; i++) {
    	for(int j = 0; j < NCLASSES; j++) {
    		printf("%.4f ", (float)hw_image_out[i*NCLASSES + j]);
    	}
    	printf("\n");
    	for(int j = 0; j < NCLASSES; j++) {
    		float val = (float)hw_image_out[i*NCLASSES + j];
			printf("%.0f ", val * 8);
		}
		printf("\n\n");
    }

    for(int k = 0; k < NPATCHES*NCLASSES; k++) {
			if (hw_image_out[k] != sw_image_out[k]) {
				err_cnt++;
				printf("%d - %.4f != %.4f\n\r",
					   k, (float)hw_image_out[k], (float)sw_image_out[k]);
			}
	}

    printf("\n%d different values\n", err_cnt);
    return err_cnt;
}
