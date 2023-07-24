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
#  *     Created: Sep 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        DOSA OSG to predict VTA performance
#  *
#  *
import os
from types import SimpleNamespace

import numpy as np
import tvm
import tvm.relay as relay
import math
import json

import gradatim.lib.singleton as dosa_singleton
from gradatim.backend.buildTools.BaseBuild import HwBuildTopVhdl
from gradatim.backend.buildTools.cFBuild1 import cFBuild1
from gradatim.backend.operatorSets.BaseOSG import BaseOSG
from gradatim.backend.devices.dosa_device import DosaHwClasses
from gradatim.lib.dosa_dtype import get_bitwidth_of_DosaDtype, DosaDtype, complete_dtype_list
from gradatim.lib.util import BrickImplTypes, rf_attainable_performance, dtype_to_bit
from gradatim.middleend.archGen.ArchBrick import ArchBrick
from gradatim.backend.operatorSets.relay_ops import op as relay_op_list
from gradatim.backend.codeGen.WrapperInterfaces import InterfaceAxisFifo, wrapper_default_interface_bitwidth
from gradatim.middleend.archGen.OperationContract import OperationContract
from gradatim.backend.operatorSets.lib.util import get_avg_util_dict_bytes_based, get_share_of_FPGA_resources
import gradatim.lib.units as units

__filedir__ = os.path.dirname(os.path.abspath(__file__))
__db_path__ = __filedir__ + '/vta_db.json'


class VtaOSG(BaseOSG):

    def __init__(self):
        super().__init__('Vta OSG', [DosaHwClasses.FPGA_xilinx, DosaHwClasses.FPGA_generic],
                         [DosaDtype.int8, DosaDtype.uint8, DosaDtype.int16, DosaDtype.int32],
                         [BrickImplTypes.ENGINE])
        self.priority = 99
        me_abs_dir = os.path.dirname(os.path.realpath(__file__))
        self.my_template_dir = None
        self.util_db = {}
        self.used_config = 'pynq_1x16_i8w8a32_15_15_18_17'
        self.peak_performance_factor = 0.88  # from ReQuEST at ASPLOS18 paper
        self.pipeline_tensor_store = 1

    def _get_impl_prediction(self, op, target_hw, impl_type, custom_latency=None, max_param_dim=-1, max_input_dim=-1):
        # if impl_type != BrickImplTypes.ENGINE or \
        #         (target_hw.hw_class != DosaHwClasses.FPGA_xilinx and target_hw.hw_class != DosaHwClasses.FPGA_generic):
        #     return None
        if max_param_dim > 0:
            op_param_dim = 1
            for d in op.dims.param:
                op_param_dim *= d
            if op_param_dim > max_param_dim:
                print("[DOSA:VTA:INFO] Can't offer an implementation for {}, due to exceeded parameter size."
                      .format(repr(op)))
                return None
            op_input_dim = np.prod(op.dims.inp)
            if op_input_dim > max_input_dim:
                print("[DOSA:VTA:INFO] Can't offer an implementation for {}, due to exceeded input size."
                      .format(repr(op)))
                return None
        op_str = op.op_call.split('.')[-1]
        dtype_str = 'int8'  # default?
        if op.used_dtype != DosaDtype.UNKNOWN:
            dtype_str = repr(op.used_dtype)

        # utilization costs are constant
        base_dict = {}
        base_dict['LUTLOG'] = self.util_db[self.used_config]['LUTLOG']
        base_dict['LUTMEM'] = self.util_db[self.used_config]['LUTMEM']
        base_dict['Registers'] = self.util_db[self.used_config]['Registers']
        base_dict['BRAM'] = self.util_db[self.used_config]['BRAM']
        base_dict['DSPs'] = self.util_db[self.used_config]['DSPs']
        peak_flops = self.util_db[self.used_config]['peak_flops']
        peak_bw_Bs = self.util_db[self.used_config]['peak_bw_Bs']
        base_dtype = self.util_db[self.used_config]['dtype']

        perf_adapt_factor = 1.0
        if target_hw.clock_period_ns > self.util_db[self.used_config]['clock_period_ns']:
            # if the clock is even slower, we need to adapt the roofline
            perf_adapt_factor = self.util_db[self.used_config]['clock_period_ns'] / target_hw.clock_period_ns
        if dtype_str != base_dtype:
            dtype_factor = dtype_to_bit(dtype_str) / dtype_to_bit(base_dtype)
            perf_adapt_factor /= dtype_factor

        adapted_peak_perf = peak_flops * perf_adapt_factor * self.peak_performance_factor
        max_perf_flops = rf_attainable_performance(op.oi_engine, adapted_peak_perf, peak_bw_Bs)

        base_dict['latency_lim_per_tensor_cycl'] = 'UNKNOWN'
        if custom_latency is None:
            if op.flops == 0 or max_perf_flops < 0.000001:
                iter_hz = adapted_peak_perf
            else:
                iter_hz = max_perf_flops / op.flops
        else:
            latency_ns = custom_latency * target_hw.get_performance_dict()['fpga_clk_ns']
            iter_hz = 1 / (latency_ns * units.nanoU)

        wrapper_dict = {'LUTLOG': 0.0, 'LUTMEM': 0.0, 'Registers': 0.0, 'BRAM': 0.0, 'DSPs': 0.0}
        util_dict = {'LUTLOG': 0.0, 'LUTMEM': 0.0, 'Registers': 0.0, 'BRAM': 0.0, 'DSPs': 0.0}

        fpga_utility = target_hw.get_resource_dict()['FPGA_utility']
        base_share = get_share_of_FPGA_resources(fpga_utility, base_dict)
        wrapper_share = wrapper_dict
        proc_share = util_dict
        # proc_comp_share = (proc_share['LUTLOG'] + proc_share['DSPs']) / 2
        base_comp_share = base_share['LUTLOG']  # we know we hardly use DSPs..
        # proc_mem_share = (proc_share['LUTMEM'] + proc_share['Registers'] + proc_share['BRAM']) / 3
        base_mem_share = max(base_share['LUTMEM'], base_share['Registers'], base_share['BRAM'])
        proc_comp_share = 0
        proc_mem_share = 0
        wrapper_comp_share = 0
        wrapper_mem_share = 0
        offer = OperationContract(op, target_hw, self, BrickImplTypes.ENGINE, iter_hz, proc_comp_share, proc_mem_share,
                                  'default', wrapper_comp_share, wrapper_mem_share, proc_share, wrapper_share,
                                  base_comp_share, base_mem_share, base_share, adapted_peak_perf, peak_bw_Bs)
        return offer

    def _get_dyn_costs(self, contract, add_brick, target_hw):
        # rough calculation
        combined_iter_hz = contract.iter_hz
        num_ops = contract.num_ops
        for op in add_brick.local_op_iter_gen():
            num_ops += 1
            op_c = self.annotate_op(op, target_hw, BrickImplTypes.ENGINE, dont_annotate=True)
            combined_iter_hz += op_c.iter_hz
        avg_iter_hz = combined_iter_hz/num_ops  # to get a rough estimate of "iter_hz" roof
        max_iter_hz = avg_iter_hz/num_ops  # single threaded mode
        return 0.0, 0.0, max_iter_hz

    def init(self, dosa_hw_classes_dict, priority_internal):
        with open(__db_path__, 'r') as infile:
            util_data = json.load(infile)
        self.util_db = util_data
        self.priority_internal = priority_internal
        self.select_dosa_hw_types(dosa_hw_classes_dict)
        for e in self.relay2osg['nn']:
            # actually, VTA should support all
            self.relay2osg['nn'][e] = self._parse_all, \
                                      lambda op, thw, it: self._get_impl_prediction(op, thw, it)
        for e in self.relay2osg:
            if type(self.relay2osg[e]) == dict:
                continue
            # actually, VTA should support all
            self.relay2osg[e] = self._parse_all, \
                                      lambda op, thw, it: self._get_impl_prediction(op, thw, it)

    def build_block(self, arch_block, build_tool, selected_contracts):
        print("[DOSA:Build:ERROR] TIPS OSG was asked to build a streaming block, but it can't. IGNORING.")
        return -1

    def build_container(self, container, build_tool, selected_contracts):
        assert isinstance(build_tool, HwBuildTopVhdl)
        arch_block = container.block_ref
        used_dir_path = build_tool.add_ip_dir(arch_block.block_uuid)
        print("[DOSA:Build:ERROR] NOT YET IMPLEMENTED. IGNORING.")
        return -1

    def _parse_all(self, op, opcode, cur_addr, next_op=None):
        # TODO...
        prog = None
        data_string = []
        consumed_next_op = False
        return prog, data_string, consumed_next_op

