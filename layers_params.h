//Arguments that change in the templates

//-------LAYER 1--------//
//Input: 9,9,200
//Number of filters: 32
//Kernel size: 1
#define X1 9
#define Y1 9
#define Z1 200
#define NF1 32
#define K1 1
#define LAYER1_WEIGHTS (NF1*Z1*K1*K1)
#define INPUT1_MEM_SIZE (X1*Y1*Z1)
#define OUT1_MEM_SIZE (X1*Y1*NF1)
#define SFEI1 0
#define SFEO1 0
#define SFEW1 3


//-------LAYER 2--------//
//Input: 9,9,32
//Number of filters: 40
//Kernel size: 1
#define X2 9
#define Y2 9
#define Z2 32
#define NF2 40
#define K2 1
#define LAYER2_WEIGHTS (NF2*Z2*K2*K2)
#define INPUT2_MEM_SIZE (X2*Y2*Z2)
#define OUT2_MEM_SIZE (X3*Y3*NF2)
#define SFEI2 SFEO1
#define SFEO2 1
#define SFEW2 2
//-------LAYER 3--------//
//Input: 9,9,40
//Number of filters: 40
//Kernel size: 1
#define X3 9
#define Y3 9
#define Z3 40
#define NF3 40
#define K3 1
#define LAYER3_WEIGHTS (NF3*Z3*K3*K3)
#define INPUT3_MEM_SIZE (X3*Y3*Z3)
#define OUT3_MEM_SIZE (X4*Y4*NF3)
#define SFEI3 SFEO2
#define SFEO3 1
#define SFEW3 1
//-------LAYER 4--------//
//Input: 9,9,40
//Number of filters: 160
//Kernel size: 1
#define X4 9
#define Y4 9
#define Z4 40
#define NF4 160
#define K4 1
#define LAYER4_WEIGHTS (NF4*Z4*K4*K4)
#define INPUT4_MEM_SIZE (X4*Y4*Z4)
#define OUT4_MEM_SIZE (X5*Y5*NF4)
#define SFEI4 SFEO3
#define SFEO4 -1
#define SFEW4 1
#define SFEBLK1 -1



//-------LAYER 5--------//
//Input: 9,9,160
//Number of filters: 48
//Kernel size: 1
#define X5 9
#define Y5 9
#define Z5 160
#define NF5 48
#define K5 1
#define LAYER5_WEIGHTS (NF5*Z5*K5*K5)
#define INPUT5_MEM_SIZE (X5*Y5*Z5)
#define OUT5_MEM_SIZE (X6*Y6*NF5)
#define SFEI5 SFEBLK1
#define SFEO5 0
#define SFEW5 3
//-------LAYER 6--------//
//Input: 9,9,48
//Number of filters: 48
//Kernel size: 1
#define X6 9
#define Y6 9
#define Z6 48
#define NF6 48
#define K6 1
#define LAYER6_WEIGHTS (NF6*Z6*K6*K6)
#define INPUT6_MEM_SIZE (X6*Y6*Z6)
#define OUT6_MEM_SIZE (X7*Y7*NF6)
#define SFEI6 SFEO5
#define SFEO6 0
#define SFEW6 1
//-------LAYER 7--------//
//Input: 9,9,48
//Number of filters: 192
//Kernel size: 1
#define X7 9
#define Y7 9
#define Z7 48
#define NF7 192
#define K7 1
#define LAYER7_WEIGHTS (NF7*Z7*K7*K7)
#define INPUT7_MEM_SIZE (X7*Y7*Z7)
#define OUT7_MEM_SIZE (X8*Y8*NF7)
#define SFEI7 SFEO6
#define SFEO7 0
#define SFEW7 2
#define SFEBLK2 -1



//-------LAYER 8--------//
//Input: 9,9,192
//Number of filters: 64
//Kernel size: 1
#define X8 9
#define Y8 9
#define Z8 192
#define NF8 64
#define K8 1
#define LAYER8_WEIGHTS (NF8*Z8*K8*K8)
#define INPUT8_MEM_SIZE (X8*Y8*Z8)
#define OUT8_MEM_SIZE (X9*Y9*NF8)
#define SFEI8 SFEBLK2
#define SFEO8 0
#define SFEW8 3
//-------LAYER 9--------//
//Input: 9,9,64
//Number of filters: 72
//Kernel size: 1
#define X9 9
#define Y9 9
#define Z9 64
#define NF9 72
#define K9 1
#define LAYER9_WEIGHTS (NF9*Z9*K9*K9)
#define INPUT9_MEM_SIZE (X9*Y9*Z9)
#define OUT9_MEM_SIZE (X10*Y10*NF9)
#define SFEI9 SFEO8
#define SFEO9 0
#define SFEW9 2
//-------LAYER 10--------//
//Input: 9,9,72
//Number of filters: 240
//Kernel size: 1
#define X10 9
#define Y10 9
#define Z10 72
#define NF10 240
#define K10 1
#define LAYER10_WEIGHTS (NF10*Z10*K10*K10)
#define INPUT10_MEM_SIZE (X10*Y10*Z10)
#define OUT10_MEM_SIZE (X11*Y11*NF10)
#define SFEI10 SFEO9
#define SFEO10 -1
#define SFEW10 2
#define SFEBLK3 -1



//-------LAYER 11--------//
//Input: 9,9,240
//Number of filters: 64
//Kernel size: 1
#define X11 9
#define Y11 9
#define Z11 240
#define NF11 64
#define K11 1
#define LAYER11_WEIGHTS (NF11*Z11*K11*K11)
#define INPUT11_MEM_SIZE (X11*Y11*Z11)
#define OUT11_MEM_SIZE (X12*Y12*NF11)
#define SFEI11 SFEBLK3
#define SFEO11 -1
#define SFEW11 4
//-------LAYER 12--------//
//Input: 9,9,64
//Number of filters: 72
//Kernel size: 2
#define X12 9
#define Y12 9
#define Z12 64
#define NF12 72
#define K12 2
#define LAYER12_WEIGHTS (NF12*Z12*K12*K12)
#define INPUT12_MEM_SIZE (X12*Y12*Z12)
#define OUT12_MEM_SIZE (X13*Y13*NF12)
#define SFEI12 SFEO11
#define SFEO12 0
#define SFEW12 3
//-------LAYER 13--------//
//Input: 4,4,,72
//Number of filters: 304
//Kernel size: 1
#define X13 4
#define Y13 4
#define Z13 72
#define NF13 304
#define K13 1
#define LAYER13_WEIGHTS (NF13*Z13*K13*K13)
#define INPUT13_MEM_SIZE (X13*Y13*Z13)
#define OUT13_MEM_SIZE (X14*Y14*NF13)
#define SFEI13 SFEO12
#define SFEO13 -1
#define SFEW13 2
//--------Downsampling 1--------//
//Input: 9,9,240
//Output: 4,4,240
//Kernel size: 2
#define XDS1 4
#define YDS1 4
#define KDS1 2
#define SFEDS1 -1
#define SFEBLK4 -1



//-------LAYER 14--------//
//Input: 4,4,304
//Number of filters: 88
//Kernel size: 1
#define X14 4
#define Y14 4
#define Z14 304
#define NF14 88
#define K14 1
#define LAYER14_WEIGHTS (NF14*Z14*K14*K14)
#define INPUT14_MEM_SIZE (X14*Y14*Z14)
#define OUT14_MEM_SIZE (X15*Y15*NF14)
#define SFEI14 SFEBLK4
#define SFEO14 -1
#define SFEW14 4
//-------LAYER 15--------//
//Input: 4,4,88
//Number of filters: 88
//Kernel size: 1
#define X15 4
#define Y15 4
#define Z15 88
#define NF15 88
#define K15 1
#define LAYER15_WEIGHTS (NF15*Z15*K15*K15)
#define INPUT15_MEM_SIZE (X15*Y15*Z15)
#define OUT15_MEM_SIZE (X16*Y16*NF15)
#define SFEI15 SFEO14
#define SFEO15 0
#define SFEW15 2
//-------LAYER 16--------//
//Input: 4,4,88
//Number of filters: 352
//Kernel size: 1
#define X16 4
#define Y16 4
#define Z16 88
#define NF16 352
#define K16 1
#define LAYER16_WEIGHTS (NF16*Z16*K16*K16)
#define INPUT16_MEM_SIZE (X16*Y16*Z16)
#define OUT16_MEM_SIZE (X17*Y17*NF16)
#define SFEI16 SFEO15
#define SFEO16 -1
#define SFEW16 2
#define SFEBLK5 -1



//-------LAYER 17--------//
//Input: 4,4,352
//Number of filters: 96
//Kernel size: 1
#define X17 4
#define Y17 4
#define Z17 352
#define NF17 96
#define K17 1
#define LAYER17_WEIGHTS (NF17*Z17*K17*K17)
#define INPUT17_MEM_SIZE (X17*Y17*Z17)
#define OUT17_MEM_SIZE (X18*Y18*NF17)
#define SFEI17 SFEBLK5
#define SFEO17 -1
#define SFEW17 4
//-------LAYER 18--------//
//Input: 4,4,96
//Number of filters: 96
//Kernel size: 1
#define X18 4
#define Y18 4
#define Z18 96
#define NF18 96
#define K18 1
#define LAYER18_WEIGHTS (NF18*Z18*K18*K18)
#define INPUT18_MEM_SIZE (X18*Y18*Z18)
#define OUT18_MEM_SIZE (X19*Y19*NF18)
#define SFEI18 SFEO17
#define SFEO18 0
#define SFEW18 2
//-------LAYER 19--------//
//Input: 4,4,96
//Number of filters: 400
//Kernel size: 1
#define X19 4
#define Y19 4
#define Z19 96
#define NF19 400
#define K19 1
#define LAYER19_WEIGHTS (NF19*Z19*K19*K19)
#define INPUT19_MEM_SIZE (X19*Y19*Z19)
#define OUT19_MEM_SIZE (X20*Y20*NF19)
#define SFEI19 SFEO18
#define SFEO19 -1
#define SFEW19 2
#define SFEBLK6 -1



//-------LAYER 20--------//
//Input: 4,4,400
//Number of filters: 96
//Kernel size: 1
#define X20 4
#define Y20 4
#define Z20 400
#define NF20 96
#define K20 1
#define LAYER20_WEIGHTS (NF20*Z20*K20*K20)
#define INPUT20_MEM_SIZE (X20*Y20*Z20)
#define OUT20_MEM_SIZE (X21*Y21*NF20)
#define SFEI20 SFEBLK6
#define SFEO20 -1
#define SFEW20 4
//-------LAYER 21--------//
//Input: 4,4,96
//Number of filters: 104
//Kernel size: 2
#define X21 4
#define Y21 4
#define Z21 96
#define NF21 104
#define K21 2
#define LAYER21_WEIGHTS (NF21*Z21*K21*K21)
#define INPUT21_MEM_SIZE (X21*Y21*Z21)
#define OUT21_MEM_SIZE (X22*Y22*NF21)
#define SFEI21 SFEO20
#define SFEO21 -1
#define SFEW21 3
//-------LAYER 22--------//
//Input: 2,2,104
//Number of filters: 440
//Kernel size: 1
#define X22 2
#define Y22 2
#define Z22 104
#define NF22 440
#define K22 1
#define LAYER22_WEIGHTS (NF22*Z22*K22*K22)
#define INPUT22_MEM_SIZE (X22*Y22*Z22)
#define OUT22_MEM_SIZE (X23*Y23*NF22)
#define SFEI22 SFEO21
#define SFEO22 -1
#define SFEW22 2
//--------Downsampling 2--------//
//Input: 4,4,200
//Output: 2,2,200
//Kernel size: 2
#define XDS2 2
#define YDS2 2
#define KDS2 2
#define SFEDS2 -1
#define SFEBLK7 -2



//-------LAYER 23--------//
//Input: 2,2,440
//Number of filters: 120
//Kernel size: 1
#define X23 2
#define Y23 2
#define Z23 440
#define NF23 120
#define K23 1
#define LAYER23_WEIGHTS (NF23*Z23*K23*K23)
#define INPUT23_MEM_SIZE (X23*Y23*Z23)
#define OUT23_MEM_SIZE (X24*Y24*NF23)
#define SFEI23 SFEBLK7
#define SFEO23 -1
#define SFEW23 4
//-------LAYER 24--------//
//Input: 2,2,120
//Number of filters: 120
//Kernel size: 1
#define X24 2
#define Y24 2
#define Z24 120
#define NF24 120
#define K24 1
#define LAYER24_WEIGHTS (NF24*Z24*K24*K24)
#define INPUT24_MEM_SIZE (X24*Y24*Z24)
#define OUT24_MEM_SIZE (X25*Y25*NF24)
#define SFEI24 SFEO23
#define SFEO24 0
#define SFEW24 2
//-------LAYER 25--------//
//Input: 2,2,120
//Number of filters: 480
//Kernel size: 1
#define X25 2
#define Y25 2
#define Z25 120
#define NF25 480
#define K25 1
#define LAYER25_WEIGHTS (NF25*Z25*K25*K25)
#define INPUT25_MEM_SIZE (X25*Y25*Z25)
#define OUT25_MEM_SIZE (X26*Y26*NF25)
#define SFEI25 SFEO24
#define SFEO25 0
#define SFEW25 3
#define SFEBLK8 -2



//-------LAYER 26--------//
//Input: 2,2,480
//Number of filters: 128
//Kernel size: 1
#define X26 2
#define Y26 2
#define Z26 480
#define NF26 128
#define K26 1
#define LAYER26_WEIGHTS (NF26*Z26*K26*K26)
#define INPUT26_MEM_SIZE (X26*Y26*Z26)
#define OUT26_MEM_SIZE (X27*Y27*NF26)
#define SFEI26 SFEBLK8
#define SFEO26 -1
#define SFEW26 4
//-------LAYER 27--------//
//Input: 2,2,128
//Number of filters: 128
//Kernel size: 1
#define X27 2
#define Y27 2
#define Z27 128
#define NF27 128
#define K27 1
#define LAYER27_WEIGHTS (NF27*Z27*K27*K27)
#define INPUT27_MEM_SIZE (X27*Y27*Z27)
#define OUT27_MEM_SIZE (X28*Y28*NF27)
#define SFEI27 SFEO26
#define SFEO27 0
#define SFEW27 3
//-------LAYER 28--------//
//Input: 2,2,128
//Number of filters: 528
//Kernel size: 1
#define X28 2
#define Y28 2
#define Z28 128
#define NF28 528
#define K28 1
#define LAYER28_WEIGHTS (NF28*Z28*K28*K28)
#define INPUT28_MEM_SIZE (X28*Y28*Z28)
#define OUT28_MEM_SIZE (X28*Y28*NF28)
#define SFEI28 SFEO27
#define SFEO28 0
#define SFEW28 3
#define SFEBLK9 -2

//----------Final Relu------//
#define SFE_RELU_I SFEBLK9
#define SFE_RELU_O 0

//----------Final Downsampling------//
#define XDS3 1
#define YDS3 1
#define KDS3 2
#define SFEDS3 0


//-------Fully Connected---------//
#define SFEW_FC 3
#define SFEB_FC 3
#define SFEO_FC 3

