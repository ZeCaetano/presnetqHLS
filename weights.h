#include "simple_conv.h"

quant_t weights_l1[1376] = {-1, -1, 5, -5, -5, -3, 1, -7, -7, 3, 6, -1, -1, 5, -5, -3, -3, 1, -7, 3, 3, 6, -1, 5, 5, -5, -3, 1, 1, -7, 3, 6, 6, -1, 5, -5, -5, -3, 1, -7, -7, 3, 6, -2, -2, 0, -6, -4, -4, 2, -8, -8, 4, 7, -2, -2, 0, -6, -4, -4, 2, -8, 4, 4, 7, -2, 0, 0, -6, -4, 2, 2, -8, 4, 7, 7, -2, 0, -6, -6, -4, 2, -8, -8, 4, 7, -3, -3, 1, -7, 3, 3, 6, -1, 5, 5, -5, -3, -3, 1, -7, 3, 3, 6, -1, 5, 5, -5, -3, 1, 1, -7, 3, 6, 6, -1, 5, -5, -5, -3, 1, -7, -7, 3, 6, -1, -1, 5, -5, -4, -4, 2, -8, 4, 4, 7, -2, 0, 0, -6, -4, 2, 2, -8, 4, 4, 7, -2, 0, 0, -6, -4, 2, 2, -8, 4, 7, 7, -2, 0, -6, -6, -4, 2, -8, -8, 4, 7, -2, -2, 0, -6, 3, 3, 6, -1, 5, 5, -5, -3, 1, 1, -7, 3, 6, 6, -1, 5, -5, -5, -3, 1, 1, -7, 3, 6, 6, -1, 5, -5, -5, -3, 1, -7, -7, 3, 6, -1, -1, 5, -5, -3, -3, 1, -7, 4, 4, 7, -2, 0, 0, -6, -4, 2, 2, -8, 4, 7, 7, -2, 0, -6, -6, -4, 2, -8, -8, 4, 7, 7, -2, 0, -6, -6, -4, 2, -8, -8, 4, 7, -2, -2, 0, -6, -4, -4, 2, -8, 5, 5, -5, -3, 1, 1, -7, 3, 6, 6, -1, 5, -5, -5, -3, 1, -7, -7, 3, 6, -1, -1, 5, -5, -3, -3, 1, -7, -7, 3, 6, -1, -1, 5, -5, -3, -3, 1, -7, 3, 3, 6, -1, 0, 0, -6, -4, 2, 2, -8, 4, 7, 7, -2, 0, -6, -6, -4, 2, -8, -8, 4, 7, -2, -2, 0, -6, -4, -4, 2, -8, 4, 4, 7, -2, -2, 0, -6, -4, -4, 2, -8, 4, 4, 7, -2, 1, 1, -7, 3, 6, 6, -1, 5, -5, -5, -3, 1, -7, -7, 3, 6, -1, -1, 5, -5, -3, -3, 1, -7, 3, 3, 6, -1, 5, 5, -5, -3, 1, 1, -7, 3, 3, 6, -1, 5, 5, -5, -3, 2, 2, -8, 4, 7, 7, -2, 0, -6, -6, -4, 2, -8, -8, 4, 7, -2, -2, 0, -6, -4, -4, 2, -8, 4, 4, 7, -2, 0, 0, -6, -4, 2, 2, -8, 4, 7, 7, -2, 0, 0, -6, -4, 6, 6, -1, 5, -5, -5, -3, 1, -7, -7, 3, 6, -1, -1, 5, -5, -3, -3, 1, -7, 3, 3, 6, -1, 5, 5, -5, -3, 1, 1, -7, 3, 6, 6, -1, 5, -5, -5, -3, 1, -7, -7, 3, 7, -2, -2, 0, -6, -6, -4, 2, -8, -8, 4, 7, -2, -2, 0, -6, -4, -4, 2, -8, 4, 4, 7, -2, 0, 0, -6, -4, 2, 2, -8, 4, 7, 7, -2, 0, -6, -6, -4, 2, -8, -8, 4, -5, -3, -3, 1, -7, 3, 3, 6, -1, -1, 5, -5, -3, -3, 1, -7, 3, 3, 6, -1, 5, 5, -5, -3, 1, 1, -7, 3, 6, 6, -1, 5, -5, -5, -3, 1, -7, -7, 3, 6, -1, -1, 5, -6, -4, -4, 2, -8, 4, 4, 7, -2, 0, 0, -6, -4, -4, 2, -8, 4, 4, 7, -2, 0, 0, -6, -4, 2, 2, -8, 4, 7, 7, -2, 0, -6, -6, -4, 2, -8, -8, 4, 7, -2, -2, 0, -7, 3, 3, 6, -1, 5, 5, -5, -3, 1, 1, -7, 3, 6, 6, -1, 5, 5, -5, -3, 1, 1, -7, 3, 6, 6, -1, 5, -5, -5, -3, 1, -7, -7, 3, 6, -1, -1, 5, -5, -3, -3, 1, -8, 4, 4, 7, -2, 0, 0, -6, -4, 2, 2, -8, 4, 7, 7, -2, 0, -6, -6, -4, 2, 2, -8, 4, 7, 7, -2, 0, -6, -6, -4, 2, -8, -8, 4, 7, -2, -2, 0, -6, -4, -4, 2, -1, 5, 5, -5, -3, 1, 1, -7, 3, 6, 6, -1, 5, -5, -5, -3, 1, -7, -7, 3, 6, -1, -1, 5, -5, -5, -3, 1, -7, -7, 3, 6, -1, -1, 5, -5, -3, -3, 1, -7, 3, 3, 6, -2, 0, 0, -6, -4, 2, 2, -8, 4, 7, 7, -2, 0, -6, -6, -4, 2, -8, -8, 4, 7, -2, -2, 0, -6, -4, -4, 2, -8, -8, 4, 7, -2, -2, 0, -6, -4, -4, 2, -8, 4, 4, 7, -3, 1, 1, -7, 3, 6, 6, -1, 5, -5, -5, -3, 1, -7, -7, 3, 6, -1, -1, 5, -5, -3, -3, 1, -7, 3, 3, 6, -1, 5, 5, -5, -3, -3, 1, -7, 3, 3, 6, -1, 5, 5, -5, -4, 2, 2, -8, 4, 7, 7, -2, 0, -6, -6, -4, 2, -8, -8, 4, 7, -2, -2, 0, -6, -4, -4, 2, -8, 4, 4, 7, -2, 0, 0, -6, -4, 2, 2, -8, 4, 4, 7, -2, 0, 0, -6, 3, 6, 6, -1, 5, -5, -5, -3, 1, -7, -7, 3, 6, -1, -1, 5, -5, -3, -3, 1, -7, 3, 3, 6, -1, 5, 5, -5, -3, 1, 1, -7, 3, 6, 6, -1, 5, -5, -5, -3, 1, 1, -7, 4, 7, 7, -2, 0, -6, -6, -4, 2, -8, -8, 4, 7, -2, -2, 0, -6, -4, -4, 2, -8, 4, 4, 7, -2, 0, 0, -6, -4, 2, 2, -8, 4, 7, 7, -2, 0, -6, -6, -4, 2, -8, -8, 5, -5, -3, -3, 1, -7, -7, 3, 6, -1, -1, 5, -5, -3, -3, 1, -7, 3, 3, 6, -1, 5, 5, -5, -3, 1, 1, -7, 3, 6, 6, -1, 5, -5, -5, -3, 1, -7, -7, 3, 6, -1, -1, 0, -6, -4, -4, 2, -8, 4, 4, 7, -2, -2, 0, -6, -4, -4, 2, -8, 4, 4, 7, -2, 0, 0, -6, -4, 2, 2, -8, 4, 7, 7, -2, 0, -6, -6, -4, 2, -8, -8, 4, 7, -2, -2, 1, -7, 3, 3, 6, -1, 5, 5, -5, -3, 1, 1, -7, 3, 3, 6, -1, 5, 5, -5, -3, 1, 1, -7, 3, 6, 6, -1, 5, -5, -5, -3, 1, -7, -7, 3, 6, -1, -1, 5, -5, -3, -3, 2, -8, 4, 4, 7, -2, 0, 0, -6, -4, 2, 2, -8, 4, 7, 7, -2, 0, 0, -6, -4, 2, 2, -8, 4, 7, 7, -2, 0, -6, -6, -4, 2, -8, -8, 4, 7, -2, -2, 0, -6, -4, -4, 6, -1, 5, 5, -5, -3, 1, 1, -7, 3, 6, 6, -1, 5, -5, -5, -3, 1, -7, -7, 3, 6, 6, -1, 5, -5, -5, -3, 1, -7, -7, 3, 6, -1, -1, 5, -5, -3, -3, 1, -7, 3, 3, 7, -2, 0, 0, -6, -4, 2, 2, -8, 4, 7, 7, -2, 0, -6, -6, -4, 2, -8, -8, 4, 7, -2, -2, 0, -6, -6, -4, 2, -8, -8, 4, 7, -2, -2, 0, -6, -4, -4, 2, -8, 4, 4, -5, -3, 1, 1, -7, 3, 6, 6, -1, 5, -5, -5, -3, 1, -7, -7, 3, 6, -1, -1, 5, -5, -3, -3, 1, -7, 3, 3, 6, -1, -1, 5, -5, -3, -3, 1, -7, 3, 3, 6, -1, 5, 5, -6, -4, 2, 2, -8, 4, 7, 7, -2, 0, -6, -6, -4, 2, -8, -8, 4, 7, -2, -2, 0, -6, -4, -4, 2, -8, 4, 4, 7, -2, 0, 0, -6, -4, -4, 2, -8, 4, 4, 7, -2, 0, 0, -7, 3, 6, 6, -1, 5, -5, -5, -3, 1, -7, -7, 3, 6, -1, -1, 5, -5, -3, -3, 1, -7, 3, 3, 6, -1, 5, 5, -5, -3, 1, 1, -7, 3, 6, 6, -1, 5, 5, -5, -3, 1, 1, -8, 4, 7, 7, -2, 0, -6, -6, -4, 2, -8, -8, 4, 7, -2, -2, 0, -6, -4, -4, 2, -8, 4, 4, 7, -2, 0, 0, -6, -4, 2, 2, -8, 4, 7, 7, -2, 0, -6, -6, -4, 2, 2};
quant_t weights_l2[1849] = {4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1};