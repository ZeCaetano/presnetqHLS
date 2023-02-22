#include "simple_conv.h"

quant_reshp weights_l1[192] = {88329711, -1985251807, 88329711, -1985251807, 88329711, -1985251807, 273956062, -124078238, 273956062, -124078238, 273956062, -124078238, 553993165, -276190346, 553993165, -276190346, 553993165, -276190346, 1645237308, -554132809, 1645237308, -554132809, 1645237308, -554132809, 1981875523, -839939669, 1981875523, -839939669, 1981875523, -839939669, -1218310060, 1021245594, -1218310060, 1021245594, -1218310060, 1021245594, -1418321659, 1137569673, -1418321659, 1137569673, -1418321659, 1137569673, -1699257840, 1413275384, -1699257840, 1413275384, -1699257840, 1413275384, -1985251807, 88329711, -1985251807, 88329711, -1985251807, 88329711, -124078238, 273956062, -124078238, 273956062, -124078238, 273956062, -276190346, 553993165, -276190346, 553993165, -276190346, 553993165, -554132809, 1645237308, -554132809, 1645237308, -554132809, 1645237308, -839939669, 1981875523, -839939669, 1981875523, -839939669, 1981875523, 1021245594, -1218310060, 1021245594, -1218310060, 1021245594, -1218310060, 1137569673, -1418321659, 1137569673, -1418321659, 1137569673, -1418321659, 1413275384, -1699257840, 1413275384, -1699257840, 1413275384, -1699257840, 88329711, -1985251807, 88329711, -1985251807, 88329711, -1985251807, 273956062, -124078238, 273956062, -124078238, 273956062, -124078238, 553993165, -276190346, 553993165, -276190346, 553993165, -276190346, 1645237308, -554132809, 1645237308, -554132809, 1645237308, -554132809, 1981875523, -839939669, 1981875523, -839939669, 1981875523, -839939669, -1218310060, 1021245594, -1218310060, 1021245594, -1218310060, 1021245594, -1418321659, 1137569673, -1418321659, 1137569673, -1418321659, 1137569673, -1699257840, 1413275384, -1699257840, 1413275384, -1699257840, 1413275384, -1985251807, 88329711, -1985251807, 88329711, -1985251807, 88329711, -124078238, 273956062, -124078238, 273956062, -124078238, 273956062, -276190346, 553993165, -276190346, 553993165, -276190346, 553993165, -554132809, 1645237308, -554132809, 1645237308, -554132809, 1645237308, -839939669, 1981875523, -839939669, 1981875523, -839939669, 1981875523, 1021245594, -1218310060, 1021245594, -1218310060, 1021245594, -1218310060, 1137569673, -1418321659, 1137569673, -1418321659, 1137569673, -1418321659, 1413275384, -1699257840, 1413275384, -1699257840, 1413275384, -1699257840};
quant_reshp weights_l2[1536] = {269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 269133140, -1986842079, -268994968, 1270664670, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, -1825431280, -143234935, -1127358737, 178607179, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1825431280, -143234935, -1127358737, 178607179, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1986562451, -841027593, -1521202244, 841027594, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, 1986562451, -841027593, -1521202244, 841027594, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -554698890, 1414249677, 554699429, 1753846578, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -554698890, 1414249677, 554699429, 1753846578, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, 1270664670, 269133140, -1986842079, -268994968, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1270664670, 269133140, -1986842079, -268994968, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, 178607179, -1825431280, -143234935, -1127358737, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 178607179, -1825431280, -143234935, -1127358737, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 841027594, 1986562451, -841027593, -1521202244, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, 841027594, 1986562451, -841027593, -1521202244, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, 1753846578, -554698890, 1414249677, 554699429, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1753846578, -554698890, 1414249677, 554699429, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -268994968, 1270664670, 269133140, -1986842079, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -268994968, 1270664670, 269133140, -1986842079, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -1127358737, 178607179, -1825431280, -143234935, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, -1127358737, 178607179, -1825431280, -143234935, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, -1521202244, 841027594, 1986562451, -841027593, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, -1521202244, 841027594, 1986562451, -841027593, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 554699429, 1753846578, -554698890, 1414249677, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 554699429, 1753846578, -554698890, 1414249677, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1986842079, -268994968, 1270664670, 269133140, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, -1986842079, -268994968, 1270664670, 269133140, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, -143234935, -1127358737, 178607179, -1825431280, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -143234935, -1127358737, 178607179, -1825431280, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -841027593, -1521202244, 841027594, 1986562451, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -841027593, -1521202244, 841027594, 1986562451, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1414249677, 554699429, 1753846578, -554698890, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, 1414249677, 554699429, 1753846578, -554698890, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, 1414249677, 554699429, 1753846578, -554698890, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, -1521202244, 841027594, 1986562451, -841027593, -1521202244, 841027594, 1986562451, -841027593, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 269133140, -1986842079, -268994968, 1270664670, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 178607179, -1825431280, -143234935, -1127358737, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670, 269133140, -1986842079, -268994968, 1270664670};
quant_reshp weights_l3[1344] = {-716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -716212599, -484373828, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -850069656, -1640886101, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, -1126871178, -1981603574, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, -1412606729, 1755197968, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, 180147535, 1988748065, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 279694676, -144138702, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 554351829, 1333168611, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318, 839953357, 1425500318};
