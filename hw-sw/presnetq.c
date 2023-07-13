#include <stdio.h>

#include "presnetq.h"

#include <xil_mmu.h>
#include <assert.h>


volatile unsigned int *ch_images; /* images data region */
volatile int *fc_out;             /* output of last fully connected layer*/


XAxiDma AxiDma0;

void init_memory() {
    /* Check if memory reserved for loading files is enough */
    assert(MEM_BIN_IMAGES >= MEM_CH_IMAGES);

    ch_images = (unsigned int *) MEM_BASE_ADDR;
    fc_out = (int*) ((unsigned char*) ch_images + MEM_BIN_IMAGES);

//    *last = 0;
    for(int i = 0; i < NCLASSES*NPATCHES; i++) {
    	fc_out[i] = 3;
    }
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

int hw_execution(){
	int Status;
	volatile unsigned int *TxBufferPtr;
	volatile int *RxBufferPtr;
	printf("#Hardware Execution: \n");
	int *output = (int*) fc_out;
	unsigned int *input = (unsigned int*) ch_images;
//	Xil_DCacheDisable();



	Xil_DCacheFlushRange((INTPTR)(output), (unsigned)(4*NCLASSES*NPATCHES));

	double t_start = xilGetMilliseconds();

	RxBufferPtr = (volatile int *) output;
	Status = XAxiDma_SimpleTransfer(&AxiDma0, (UINTPTR) RxBufferPtr, 4*NCLASSES*NPATCHES, XAXIDMA_DEVICE_TO_DMA);
	if (Status != XST_SUCCESS) return XST_FAILURE;

	for(int i = 0; i < NPATCHES; i++) {
		input = (unsigned int*) ((unsigned char*) ch_images + (i*MEM_BIN_SINGLE_IMAGE));
		if(i == NPATCHES-1)
			input[0] = 1; //last flag
		else
			input[0] = 0;
		Xil_DCacheFlushRange((INTPTR)(input), (unsigned)(4*(INPUT_SIZE+1)));
		TxBufferPtr = (volatile unsigned int *) input;
		Status = XAxiDma_SimpleTransfer(&AxiDma0, (UINTPTR) TxBufferPtr, 4*(INPUT_SIZE+1), XAXIDMA_DMA_TO_DEVICE);
		if (Status != XST_SUCCESS) return XST_FAILURE;
//		printf("#Waiting for input\n");
//		if(i == NPATCHES - 1) *last = 1;
		while(XAxiDma_Busy(&AxiDma0, XAXIDMA_DMA_TO_DEVICE)){/*Wait for tx*/ }
	}
//	printf("#Waiting for output\n");
	while(XAxiDma_Busy(&AxiDma0, XAXIDMA_DEVICE_TO_DMA)){/*Wait for rx*/ }


	Xil_DCacheInvalidateRange((INTPTR)output, (unsigned)(4*NCLASSES*NPATCHES));

    double t_end = xilGetMilliseconds();

    printf("#Hardware execution took %.0f us.\n\r", t_end - t_start);
    return 0;
}

double xilGetMilliseconds() {
    XTime time;
    XTime_GetTime(&time);
    return (double) time * 1000 *1000 / COUNTS_PER_SECOND;
}

int main(int argc, char **argv) {
	int Status;
	printf("#Init memory\n");
	/* Performs memory assignment */
	init_memory();

	double start_time = 0;
	double end_time = 0;
	double exec_time = 0;

	exec_time = 0;


	printf("Init DMA\n");
	/* Init DMA in poll mode for simple transfer */
	Status = init_XAxiDma_SimplePollMode();
	if (Status != XST_SUCCESS) {
		printf("init_XAxiDma_SimplePollMode: Failed\r\n");
		exit(XST_FAILURE);
	}
	exec_time = 0;

	start_time = xilGetMilliseconds();
//	printf("%p\n", fully_connected);
	hw_execution();
	end_time = xilGetMilliseconds();
	exec_time += end_time - start_time;

	printf("#Output: \n");
	for(int i = 0; i < NPATCHES; i++){
		for(int j = 0; j < NCLASSES; j++) {
			printf("%d ", fc_out[i*NCLASSES + j]);
		}
		printf("\n\n");
	}
//	printf("#Full Hardware execution time: %.2f ms\n\r", exec_time);
	return 0;
}
