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
from dimidium.backend.devices import DosaBaseHw


network_bandwidth_gBs = 10.0/8.0
cpu_gflops = 200  # from some benchmark for Intel i7 8th gen
dram_bandwith_gBs = 80.0


class VcpuDummy(DosaBaseHw):

    def __init__(self, hw_type, name):
        super().__init__(hw_type, name)

    def get_performance_dict(self):
        ret = {'type': 'x86', 'cpu_gflops': cpu_gflops, 'bw_netw_gBs': network_bandwidth_gBs,
               'bw_dram_gBs': dram_bandwith_gBs}
        return ret

    def get_roofline_dict(self):
        ret = {'sweet_spot': 0.081}  # TODO update
        return ret

    def get_resource_dict(self):
        return

    def get_max_flops(self):
        return cpu_gflops * gigaU

    def get_comm_latency_s(self):
        return 500 * mikroU  # TODO update

