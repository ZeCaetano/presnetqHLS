#include <stdio.h>
#include "xil_cache.h"
#include "xaxidma.h"
#include "xtime_l.h"

#define DMA0_DEV_ID		XPAR_AXIDMA_0_DEVICE_ID


/* ========================== START OF TEST SET CONFIGURATION ========================== */
#define NPATCHES 1
#define NCLASSES 16                 /* number of possible classes */
#define RESHAPE_FACTOR 8

#define INPUT_HEIGHT 9             /* width of the input images */
#define INPUT_WIDTH INPUT_HEIGHT     /* height of the input images */
#define INPUT_CHANNELS 200             /* number of input channels */
#define INPUT_SIZE (INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS / RESHAPE_FACTOR)



/* ============================= START OF MEMORY ASSIGNMENT ============================ */
/* Align memory regions for PS-PL data transference purposes */
#define ALIGN_CONSTANT 0x1000                          /* ALIGN_CONSTANT = 4 kB */
#define ALIGN(MEM_REGION) (                            /* aligns memory region to ALIGN_CONSTANT if not aligned */ \
    MEM_REGION % ALIGN_CONSTANT == 0 ?                 /* if memory region is aligned */ \
    MEM_REGION :                                       /* does nothing else */ \
    (MEM_REGION / ALIGN_CONSTANT + 1) * ALIGN_CONSTANT /* aligns memory region to next ALIGN_CONSTANT */ \
    )

/* Size in bytes of reserved memory regions */
/* 8100B for each image = 48600B for all 6 images*/
#define MEM_BIN_IMAGES 0x0000BDD8
#define MEM_CH_IMAGES ALIGN(  \
    NPATCHES *                \
    INPUT_SIZE *              \
    sizeof(unsigned char)     \
    )

#define MEM_BASE_ADDR 0x10000000


/**
 * Assigns memory regions to required data.
 */
void init_memory();
/**
 * Initialize the XAxiDma device.
 */
int init_XAxiDma_SimplePollMode();

double xilGetMilliseconds();

int hw_execution(unsigned int *input, float *output);




