#  /*******************************************************************************
#   * Copyright 2019 -- 2024 IBM Corporation
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
#  *     Created: Mar 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Class of the implementation contracts offered by OSGs
#  *
#  *
from gradatim.lib.util import BrickImplTypes
from gradatim.middleend.archGen.DosaContract import DosaContract


class OperationContract(DosaContract):

    # def __init__(self, op: ArchOp, device: DosaBaseHw, osg: BaseOSG, impl_type: BrickImplTypes,
    #              iter_hz: float, comp_util_share: float, mem_util_share: float, internal_id: str):
    def __init__(self, op, device, osg, impl_type, iter_hz: float, comp_util_share: float, mem_util_share: float,
                 internal_id: str, switching_comp_share: float, switching_mem_share: float, detailed_FPGA_res=None,
                 detailed_FPGA_wrapper=None,
                 engine_base_comp_share=-1, engine_base_mem_share=-1, detailed_FPGA_engine_base=None,
                 engine_max_ops=-1, engine_limiting_bw_Bs=-1):
        super().__init__(device, osg, impl_type, iter_hz, comp_util_share, mem_util_share)
        self.op = op
        self.num_ops = 1
        # self.device = device
        # self.osg = osg
        # self.impl_type = impl_type
        # self.iter_hz = iter_hz
        # self.comp_util_share = comp_util_share
        # self.mem_util_share = mem_util_share
        self.flops_per_iter = op.flops
        self.total_bytes = op.input_bytes
        if impl_type == BrickImplTypes.ENGINE:
            self.total_bytes += op.parameter_bytes
        self.oi_iter = self.total_bytes/self.iter_hz
        self.osg_intern_id = internal_id
        self.switching_comp_share = switching_comp_share
        self.switching_mem_share = switching_mem_share
        self.detailed_FPGA_component_share = detailed_FPGA_res
        self.detailed_FPGA_wrapper_share = detailed_FPGA_wrapper
        self.detailed_FPGA_engine_base_share = detailed_FPGA_engine_base
        self.engine_base_comp_share = engine_base_comp_share
        self.engine_base_mem_share = engine_base_mem_share
        self.engine_max_ops = engine_max_ops
        self.engine_limiting_bw_Bs = engine_limiting_bw_Bs
        # TODO: add wrapper_latency? add 1-batch latency?

    def __repr__(self):
        return "OpContr({} on {} using {}/{}: {:.2f}/s, {:.2f}c%, {:.2f}m%, {})"\
            .format(self.op.op_call, self.device.name, self.osg.name, self.impl_type, self.iter_hz,
                    self.comp_util_share*100, self.mem_util_share*100, self.osg_intern_id[0:9])






