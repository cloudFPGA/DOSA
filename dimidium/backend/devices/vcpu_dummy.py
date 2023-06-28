#  /*******************************************************************************
#   * Copyright 2019 -- 2023 IBM Corporation
#   *
#   * Licensed under the Apache License, Version 2.0 (the "License");
#   * you may not use this file except in compliance with the License.
#   * You may obtain a copy of the License at
#   *
#   *     http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS,
#   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   * See the License for the specific language governing permissions and
#   * limitations under the License.
#  *******************************************************************************/
#

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

import dimidium.lib.singleton as dosa_singleton
from dimidium.lib.units import *
from dimidium.backend.devices.dosa_device import DosaHwClasses, DosaBaseHw
from dimidium.backend.buildTools.DefaultCpuBuild import DefaultCpuBuild

network_bandwidth_gBs = 10.0 / 8.0
cpu_gflops = 200  # from some benchmark for Intel i7 8th gen
dram_bandwith_gBs = 80.0
cpu_l1l2_cache_size_bytes = 8 * megaU
max_connections_per_s_single_socket = 4000  # based on experiments


class VcpuDummy(DosaBaseHw):

    def __init__(self, name):
        super().__init__(DosaHwClasses.CPU_x86, 'vCPU_x86', name, [DefaultCpuBuild])
        self.initialized = False
        self.global_main_body_tmpl_path = None
        self.global_main_head_tmpl_path = None
        self.total_bytes_hw = None
        self.total_flops_hw = None

    def _gen_numbers(self):
        if self.initialized:
            return
        self.total_flops_hw = cpu_gflops * dosa_singleton.config.dtype.dosa_kappa
        self.total_bytes_hw = cpu_l1l2_cache_size_bytes
        self.initialized = True
        return

    def get_performance_dict(self):
        self._gen_numbers()
        ret = {'type': str(self.hw_class), 'cpu_gflops': self.total_flops_hw,
               # 'gflops_limit': self.total_flops_hw,
               'bw_netw_gBs': network_bandwidth_gBs,
               'bw_dram_gBs': dram_bandwith_gBs}
        return ret

    def get_roofline_dict(self):
        self._gen_numbers()
        ret = {'sweet_spot': 2.51}
        return ret

    def get_resource_dict(self):
        self._gen_numbers()
        ret = {'total_flops': self.total_flops_hw, 'total_on_chip_memory_bytes': self.total_bytes_hw}
        return ret

    def get_max_flops(self):
        self._gen_numbers()
        return self.total_flops_hw * gigaU

    def get_comm_latency_s(self):
        self._gen_numbers()
        return 500 * mikroU  # TODO update

    def get_hw_utilization_tuple(self, flops, bake_in_params_bytes):
        self._gen_numbers()
        share_flops = float(flops / self.total_flops_hw) * dosa_singleton.config.utilization.dosa_mu_comp
        share_memory = float(bake_in_params_bytes / self.total_bytes_hw) * dosa_singleton.config.utilization.dosa_mu_mem
        return share_flops, share_memory

    def get_max_connections_per_s(self):
        return max_connections_per_s_single_socket
