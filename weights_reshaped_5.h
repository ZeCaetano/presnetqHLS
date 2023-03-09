#include "simple_conv.h"

wght_reshp weights_l1[192] = {20046, 20046, 20046, 20046, 20046, 20046, 37779, 37779, 37779, 37779, 37779, 37779, 58596, 58596, 58596, 58596, 58596, 58596, 14649, 14649, 14649, 14649, 14649, 14649, 20046, 20046, 20046, 20046, 20046, 20046, 37779, 37779, 37779, 37779, 37779, 37779, 58596, 58596, 58596, 58596, 58596, 58596, 14649, 14649, 14649, 14649, 14649, 14649, 20046, 20046, 20046, 20046, 20046, 20046, 37779, 37779, 37779, 37779, 37779, 37779, 58596, 58596, 58596, 58596, 58596, 58596, 14649, 14649, 14649, 14649, 14649, 14649, 20046, 20046, 20046, 20046, 20046, 20046, 37779, 37779, 37779, 37779, 37779, 37779, 58596, 58596, 58596, 58596, 58596, 58596, 14649, 14649, 14649, 14649, 14649, 14649, 20046, 20046, 20046, 20046, 20046, 20046, 37779, 37779, 37779, 37779, 37779, 37779, 58596, 58596, 58596, 58596, 58596, 58596, 14649, 14649, 14649, 14649, 14649, 14649, 20046, 20046, 20046, 20046, 20046, 20046, 37779, 37779, 37779, 37779, 37779, 37779, 58596, 58596, 58596, 58596, 58596, 58596, 14649, 14649, 14649, 14649, 14649, 14649, 20046, 20046, 20046, 20046, 20046, 20046, 37779, 37779, 37779, 37779, 37779, 37779, 58596, 58596, 58596, 58596, 58596, 58596, 14649, 14649, 14649, 14649, 14649, 14649, 20046, 20046, 20046, 20046, 20046, 20046, 37779, 37779, 37779, 37779, 37779, 37779, 58596, 58596, 58596, 58596, 58596, 58596, 14649, 14649, 14649, 14649, 14649, 14649};
wght_reshp weights_l2[1536] = {27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 5820, 5820, 5820, 5820, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 49515, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 48150, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585, 27585};
wght_reshp weights_l3[1344] = {10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 10023, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 51657, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 29298, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092, 40092};
