#include "simple_conv.h"

quant_t weights_l1[1376] = {-1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2};
quant_t weights_l2[1849] = {4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1, 2, 3, -7, -8, 6, 7, -1, -2, -3, -4, -5, 4, 5, -6, 0, 1};
