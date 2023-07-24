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
#  *    =============================================
#  *     Created: July 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Base class for DOSA target devices
#  *
#  *

import abc
from enum import Enum

import gradatim.lib.units as units
from gradatim.backend.buildTools.BaseBuild import BaseBuild


class DosaHwClasses(Enum):
    UNDECIDED = 0
    CPU_generic = 1
    FPGA_generic = 2
    CPU_x86 = 3
    FPGA_xilinx = 4


class DosaBaseHw(metaclass=abc.ABCMeta):

    def __init__(self, hw_class: DosaHwClasses, type_str, name, possible_builders: [BaseBuild]):
        self.hw_class = hw_class
        self.name = name
        self.type_str = type_str
        self.roof_F = 0
        self.global_main_body_tmpl_path = None
        self.global_main_head_tmpl_path = None
        self.possible_builders = possible_builders
        self.build_tool_class = None
        self.part_string = 'unknown'
        self.clock_period_ns = 1.0
        self.clock_period_s = self.clock_period_ns * 1e-9
        if len(self.possible_builders) > 0:
            self.build_tool_class = self.possible_builders[0]

    def __repr__(self):
        return "DosaHwType({}, for {})".format(self.name, self.hw_class)

    def __eq__(self, other):
        if not isinstance(other, DosaBaseHw):
            return False
        return self.hw_class == other.hw_class and self.type_str == other.type_str
        # TODO: consider also name?

    def __hash__(self):
        return hash((self.hw_class, self.name, self.type_str))

    @abc.abstractmethod
    def get_performance_dict(self):
        print("[DOSA:DEVICES:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def get_roofline_dict(self):
        print("[DOSA:DEVICES:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def get_resource_dict(self):
        print("[DOSA:DEVICES:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def get_max_flops(self):
        print("[DOSA:DEVICES:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def get_comm_latency_s(self):
        print("[DOSA:DEVICES:ERROR] NOT YET IMPLEMENTED.")

    def get_roof_F(self):
        perf_dict = self.get_performance_dict()
        if 'dsp48_gflops' in perf_dict:
            self.roof_F = perf_dict['dsp48_gflops'] * units.gigaU
        else:
            self.roof_F = perf_dict['cpu_gflops'] * units.gigaU
        return self.roof_F

    def create_build_tool(self, node_id: int) -> BaseBuild:
        assert self.build_tool_class is not None
        new_bt = self.build_tool_class("{}_node_{}".format(self.name, node_id), self)
        return new_bt

    @abc.abstractmethod
    def get_hw_utilization_tuple(self, flops, bake_in_params_bytes):
        print("[DOSA:DEVICES:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def get_max_connections_per_s(self):
        print("[DOSA:DEVICES:ERROR] NOT YET IMPLEMENTED.")


class UndecidedDosaHw(DosaBaseHw):

    def __init__(self, name):
        super().__init__(DosaHwClasses.UNDECIDED, 'dosa_undecided_hw', name, [BaseBuild])

    def get_performance_dict(self):
        ret = {'type': str(self.hw_class), 'cpu_gflops': 1, 'bw_netw_gBs': 1,
               'bw_dram_gBs': 1}
        return ret

    def get_roofline_dict(self):
        pass

    def get_resource_dict(self):
        pass

    def get_max_flops(self):
        pass

    def get_comm_latency_s(self):
        pass

    def get_hw_utilization_tuple(self, flops, bake_in_params_bytes):
        pass

    def get_max_connections_per_s(self):
        pass


placeholderHw = UndecidedDosaHw('DOSA_placeholder')

