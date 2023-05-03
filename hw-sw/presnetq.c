#include <stdio.h>

#include "presnetq.h"

#include <xil_mmu.h>
#include <assert.h>

#include "xparameters.h"

volatile unsigned int *ch_images; /* images data region */
volatile float *fc_out;            /* output of last fully connected layer*/
volatile int *do_presnet = (int *)(XPAR_SIMPLE_CONV_0_S_AXI_CONTROL_BASEADDR);
XAxiDma AxiDma0;

void init_memory() {
    /* Check if memory reserved for loading files is enough */
    assert(MEM_BIN_IMAGES >= MEM_CH_IMAGES);

    ch_images = (unsigned int *) MEM_BASE_ADDR;
    fc_out = (float*) ((unsigned char*) ch_images + MEM_BIN_IMAGES);
}


int init_XAxiDma_SimplePollMode() {
  XAxiDma_Config *CfgPtr;
  int Status;
  u16 DeviceId = DMA0_DEV_ID;


  /* Initialize the XAxiDma0 device.	 */
  CfgPtr = XAxiDma_LookupConfig(DeviceId);
  if (!CfgPtr) {
    printf("No config found for %d\r\n", DeviceId);
    return XST_FAILURE;
  }

  Status = XAxiDma_CfgInitialize(&AxiDma0, CfgPtr);
  if (Status != XST_SUCCESS) {
    printf("Initialization failed %d\r\n", Status);
    return XST_FAILURE;
  }

  if(XAxiDma_HasSg(&AxiDma0)){
    printf("Device configured as SG mode \r\n");
    return XST_FAILURE;
  }

  /* Disable interrupts, we use polling mode	 */
  XAxiDma_IntrDisable(&AxiDma0, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
  XAxiDma_IntrDisable(&AxiDma0, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

  return XST_SUCCESS;
}

int hw_execution(unsigned int *input, float *output){
	int Status;
	volatile unsigned int *TxBufferPtr;
	volatile float *RxBufferPtr;
	printf("#Hardware Execution: \n");

	for(int i = 0; i< 10; i++){
			printf("%d ", input[i]);
	}
	printf("\n");
	Xil_DCacheFlushRange((INTPTR)(input), (unsigned)(4*INPUT_SIZE));
	Xil_DCacheFlushRange((INTPTR)(output), (unsigned)(4*NCLASSES));

	double t_start = xilGetMilliseconds();
//	printf("#Send input\n");
	TxBufferPtr = (volatile unsigned int *) input;
	Status = XAxiDma_SimpleTransfer(&AxiDma0, (UINTPTR) TxBufferPtr, 4*INPUT_SIZE, XAXIDMA_DMA_TO_DEVICE);
	if (Status != XST_SUCCESS) return XST_FAILURE;
//	printf("#Waiting for input\n");
	while(XAxiDma_Busy(&AxiDma0, XAXIDMA_DMA_TO_DEVICE)){/*Wait for tx*/ }

//	printf("#Receive output\n");
	RxBufferPtr = (volatile float *) output;
	printf("%p\n", RxBufferPtr);
	Status = XAxiDma_SimpleTransfer(&AxiDma0, (UINTPTR) RxBufferPtr, 4*NCLASSES, XAXIDMA_DEVICE_TO_DMA);
	if (Status != XST_SUCCESS) return XST_FAILURE;
//	printf("#Waiting for output\n");
	while(XAxiDma_Busy(&AxiDma0, XAXIDMA_DEVICE_TO_DMA)){/*Wait for rx*/ }
//	printf("ctrl: %x\n",*do_presnet);


	Xil_DCacheInvalidateRange((INTPTR)output, (unsigned)(4*NCLASSES));

    double t_end = xilGetMilliseconds();

    printf("#Hardware execution for one image took %.0f ms.\n\r", t_end - t_start);
    return 0;
}

double xilGetMilliseconds() {
    XTime time;
    XTime_GetTime(&time);
    return (double) time * 1000 / COUNTS_PER_SECOND;
}

int main(int argc, char **argv) {
	int Status;
	printf("#Init memory\n");
	/* Performs memory assignment */
	init_memory();

	double start_time = 0;
	double end_time = 0;
	double exec_time = 0;

//	printf("Disable cache on OCM region\n");
//    // Disable cache on OCM region
//	Xil_SetTlbAttributes(0xFFFC0000,0x14de2);
	exec_time = 0;


	printf("Init DMA\n");
	/* Init DMA in poll mode for simple transfer */
	Status = init_XAxiDma_SimplePollMode();
	if (Status != XST_SUCCESS) {
		printf("init_XAxiDma_SimplePollMode: Failed\r\n");
		exit(XST_FAILURE);
	}
	printf("Start main for loop\n");
	exec_time = 0;

	for(int i = 0; i < NPATCHES; i++) {
		start_time = xilGetMilliseconds();
		*do_presnet = 1;
		unsigned int *image_in = (unsigned int *) ch_images + i*(INPUT_SIZE/RESHAPE_FACTOR);
		float *fully_connected = (float*) fc_out + i*NCLASSES;
		printf("%p\n", fully_connected);
		hw_execution(image_in, fully_connected);
		end_time = xilGetMilliseconds();
		exec_time += end_time - start_time;

	}
	printf("#Output: \n");
	for(int i = 0; i < NPATCHES; i++){
		for(int j = 0; j < NCLASSES; j++) {
			printf("%.4f ", fc_out[i*NPATCHES + j]);
		}
		printf("\n");
	}
	printf("#Execution time: %.2f ms\n\r", exec_time);
	return 0;
}
