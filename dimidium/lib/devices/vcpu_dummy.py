#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: July 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Device data for a virtualized x86 CPU
#  *        MUST IMPLEMENT: get_performance_dict(), get_roofline_dict(), get_resource_dict()
#  *

from dimidium.lib.units import *


network_bandwidth_gBs = 10.0/8.0
cpu_gflops = 200  # from some benchmark for Intel i7 8th gen

# TODO: consider creating a class/singelton?


def get_performance_dict():
    ret = {'type': 'x86', 'cpu_gflops': cpu_gflops, 'bw_netw_gBs': network_bandwidth_gBs}
    return ret


def get_roofline_dict():
    ret = {'sweet_spot': 0.081}  # TODO update
    return ret


def get_resource_dict():
    return

