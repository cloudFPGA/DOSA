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
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        DOSA OSG to execute TVM operations on CPU
#  *
#  *
from gradatim.lib import units
from gradatim.lib.util import deep_update, BrickImplTypes
from gradatim.backend.operatorSets.relay_ops import op as relay_op_list
from gradatim.backend.operatorSets.BaseOSG import BaseOSG
from gradatim.backend.devices.dosa_device import DosaHwClasses
from gradatim.middleend.archGen.ArchBrick import ArchBrick
from gradatim.middleend.archGen.OperationContract import OperationContract
from gradatim.lib.dosa_dtype import complete_dtype_list


class TvmCpuOsg(BaseOSG):

    def __init__(self):
        super().__init__('Tvm CPU OSG', [DosaHwClasses.CPU_generic, DosaHwClasses.CPU_x86], complete_dtype_list,
                         [BrickImplTypes.ENGINE, BrickImplTypes.STREAM])
        # TODO: does difference between stream and engine makes sense for CPUs?
        #  This should anyhow only be fallback?
        self.priority = 49

    def init(self, dosa_hw_classes_dict, priority_internal):
        self.priority_internal = priority_internal
        self.select_dosa_hw_types(dosa_hw_classes_dict)
        # this one will support everything
        # self.relay2osg = {x: self._generate_tvm_module for x in relay_op_list}
        self.relay2osg = deep_update(self.relay2osg, (True, self._get_contr_offer))

    def _get_contr_offer(self, op, target_hw, impl_type):
        if impl_type != BrickImplTypes.ENGINE:
            return None

        limit_bw_Bs = target_hw.get_performance_dict()['bw_dram_gBs'] * units.gigaU
        limit_flops = target_hw.get_performance_dict()['cpu_gflops'] * units.gigaU
        placeholder_contr = OperationContract(op, target_hw, self, impl_type, 1, 1.0, 1.0,
                                              "not-yet-implemented", 0.0, 0.0,
                                              engine_limiting_bw_Bs=limit_bw_Bs, engine_max_ops=limit_flops)
        return placeholder_contr

    def build_block(self, arch_block, build_tool, selected_contracts):
        pass

    def build_container(self, container, build_tool, selected_contracts):
        pass

    # def generate_brick(self, brick_node: ArchBrick):
    #     pass

    # def generate_bricks(self, brick_nodes: [ArchBrick]):
    #     # to generate subsequent bricks at once
    #     pass

    # def _generate_tvm_module(self, tvm_handle):
    #     return

    # def comm_wrap_brick(self, todo):
    #     pass

    # def estimate_flops_brick(self, brick_node: ArchBrick):
    #     pass

