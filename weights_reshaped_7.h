#include "simple_conv.h"

wght_reshp weights_l1[192] = {59294, 40569, 31207, 59294, 31207, 59294, 40569, 31207, 40569, 31207, 59294, 40569, 59294, 40569, 31207, 59294, 31207, 59294, 40569, 31207, 40569, 31207, 59294, 40569, 59294, 40569, 31207, 59294, 31207, 59294, 40569, 31207, 40569, 31207, 59294, 40569, 59294, 40569, 31207, 59294, 31207, 59294, 40569, 31207, 40569, 31207, 59294, 40569, 59294, 40569, 31207, 59294, 31207, 59294, 40569, 31207, 40569, 31207, 59294, 40569, 59294, 40569, 31207, 59294, 31207, 59294, 40569, 31207, 40569, 31207, 59294, 40569, 59294, 40569, 31207, 59294, 31207, 59294, 40569, 31207, 40569, 31207, 59294, 40569, 59294, 40569, 31207, 59294, 31207, 59294, 40569, 31207, 40569, 31207, 59294, 40569, 59294, 40569, 31207, 59294, 31207, 59294, 40569, 31207, 40569, 31207, 59294, 40569, 59294, 40569, 31207, 59294, 31207, 59294, 40569, 31207, 40569, 31207, 59294, 40569, 59294, 40569, 31207, 59294, 31207, 59294, 40569, 31207, 40569, 31207, 59294, 40569, 59294, 40569, 31207, 59294, 31207, 59294, 40569, 31207, 40569, 31207, 59294, 40569, 59294, 40569, 31207, 59294, 31207, 59294, 40569, 31207, 40569, 31207, 59294, 40569, 59294, 40569, 31207, 59294, 31207, 59294, 40569, 31207, 40569, 31207, 59294, 40569, 59294, 40569, 31207, 59294, 31207, 59294, 40569, 31207, 40569, 31207, 59294, 40569, 59294, 40569, 31207, 59294, 31207, 59294, 40569, 31207, 40569, 31207, 59294, 40569};
wght_reshp weights_l2[1536] = {54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973, 48491, 27606, 54973};
wght_reshp weights_l3[1344] = {31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569, 31207, 59294, 40569};
