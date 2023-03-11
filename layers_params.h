//Arguments that change in the templates

////------------------------------ 2 Layers: 1st k=1 s=1, 2nd k=2 s=2 --------------------------------//////////
//-------LAYER 1--------//
//Input: 9,9,32
//Number of filters: 43
//Kernel size: 1
#define X1 9
#define Y1 9
#define Z1 32
#define NF1 48
#define K1 1
#define LAYER1_WEIGHTS (NF1*Z1*K1*K1)
#define INPUT1_MEM_SIZE (X1*Y1*Z1)
#define OUT1_FM_MEM_SIZE (X2*Y2*NF1)
//-------LAYER 2--------//
//For kernel = 2, stride = 2
//Input: 9,9,43
//Number of filters: 43
//Kernel size: 2
#define X2 9
#define Y2 9
#define Z2 48
#define NF2 64
#define K2 2
#define LAYER2_WEIGHTS (NF2*Z2*K2*K2)
#define INPUT2_MEM_SIZE (X2*Y2*Z2)
#define OUT2_FM_MEM_SIZE (X3*Y3*NF2)
//-------LAYER 3--------//
//For kernel = 2, stride = 2
//Input: 4,4,43
//Number of filters: 172
//Kernel size: 2
#define X3 4
#define Y3 4
#define Z3 64
#define NF3 168
#define K3 1
#define LAYER3_WEIGHTS (Z3*NF3*K3*K3)
#define INPUT3_MEM_SIZE (X3*Y3*Z3)
#define OUT3_FM_MEM_SIZE ((X3*Y3*NF3))

#define WEIGHTS_MEM_SIZE (LAYER1_WEIGHTS+LAYER2_WEIGHTS+LAYER3_WEIGHTS)
#define OUTPUT_MEM_SIZE (X3*Y3*NF3)
//-------DOWNSAMPLING LAYER--------//
//Input: LAYER 1
//Output: 4,4,32
//Kernel size: 2
#define XDS 4
#define YDS 4
#define ZDS 32
#define KDS 2
#define OUTDS_FM_MEM_SIZE (XDS*YDS*ZDS)

////////////------------------------------ 2 Layers: 1st k=1 s=1, 1nd k=1 s=1 --------------------------------//////////
////-------LAYER 1--------//
////Input: 9,9,32
////Number of filters: 43
////Kernel size: 1
//#define X1 9
//#define Y1 9
//#define Z1 32
//#define NF1 48
//#define K1 1
//#define LAYER1_WEIGHTS (NF1*Z1*K1*K1)
//#define INPUT1_MEM_SIZE (X1*Y1*Z1)
//#define OUT1_FM_MEM_SIZE (X2*Y2*NF1)
////-------LAYER 2--------//
////For kernel = 1, stride = 1
////Input: 9,9,43
////Number of filters: 43
////Kernel size: 1
//#define X2 9
//#define Y2 9
//#define Z2 48
//#define NF2 48
//#define K2 1
//#define LAYER2_WEIGHTS (NF2*Z2*K2*K2)
//#define INPUT2_MEM_SIZE (X2*Y2*Z2)
//#define OUT2_FM_MEM_SIZE (X2*Y2*NF2)
//#define WEIGHTS_MEM_SIZE (LAYER1_WEIGHTS+LAYER2_WEIGHTS)
////-------LAYER 3--------//
////For kernel = 2, stride = 2
////Input: 4,4,43
////Number of filters: 172
////Kernel size: 2
//#define X3 9
//#define Y3 9
//#define Z3 48
//#define NF3 172
//#define K3 1
//#define LAYER3_WEIGHTS (Z3*NF3*K3*K3)
//#define OUTPUT_MEM_SIZE (X3*Y3*Z3)

////////////------------------------------ 2 Layers: 1st k=1 s=1, 1nd k=1 s=1 --------------------------------//////////
////-------LAYER 1--------//
////Input: 9,9,80
////Number of filters: 96
////Kernel size: 1
//#define X1 9
//#define Y1 9
//#define Z1 80
//#define NF1 96
//#define K1 1
//#define LAYER1_WEIGHTS (NF1*Z1*K1*K1)
//#define INPUT1_MEM_SIZE (X1*Y1*Z1)
//#define OUT1_FM_MEM_SIZE (X2*Y2*NF1)
////-------LAYER 2--------//
////For kernel = 1, stride = 1
////Input: 9,9,96
////Number of filters: 80
////Kernel size: 1
//#define X2 9
//#define Y2 9
//#define Z2 96
//#define NF2 80
//#define K2 1
//#define LAYER2_WEIGHTS (NF2*Z2*K2*K2)
//#define INPUT2_MEM_SIZE (X2*Y2*Z2)
//#define OUT2_FM_MEM_SIZE (X2*Y2*NF2)
//#define WEIGHTS_MEM_SIZE (LAYER1_WEIGHTS+LAYER2_WEIGHTS)
////-------LAYER 3--------//
////For kernel = 2, stride = 2
////Input: 4,4,80
////Number of filters: 172
////Kernel size: 2
//#define X3 9
//#define Y3 9
//#define Z3 80
//#define NF3 172
//#define K3 1
//#define LAYER3_WEIGHTS (Z3*NF3*K3*K3)
//#define OUTPUT_MEM_SIZE (X3*Y3*Z3)





////Average pool test
//#define X1 8
//#define Y1 8
//#define Z1 3
//#define X2 4
//#define Y2 4
//#define Z2 3
