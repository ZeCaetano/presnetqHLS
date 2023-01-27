#include "simple_conv.h"

quant_t weights_l1[1536] = {-1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7, -8, -1, -2, -3, -4, 3, 4, 5, 0, 1, 2, 6, 7, -5, -6, -7};
quant_t weights_l2[9216] = {4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4, 4, 5, 5, -6, -6, 0, 0, 1, 1, 2, 2, 3, 3, -7, -7, -8, -8, 6, 6, 7, 7, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, 4};