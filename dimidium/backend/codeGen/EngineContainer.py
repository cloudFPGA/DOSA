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
#  *     Created: Nov 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Class to help the generation of Engine blocks
#  *
#  *
from dimidium.backend.operatorSets.BaseOSG import placeholderOSG
from dimidium.lib.util import BrickImplTypes
from dimidium.middleend.archGen.ArchBlock import ArchBlock
from dimidium.middleend.archGen.ArchOp import ArchOp


class EngineContainer(object):

    def __init__(self, arch_block):
        assert arch_block.block_impl_type == BrickImplTypes.ENGINE
        self.block_ref = arch_block
        self.ops = {}
        self.resource_savings = 0
        self.init_container()

    def __repr__(self):
        ops_list = [k for k in self.ops.keys()]
        ret = "EngineContainer(block_ref: {}, savings: {}, ops: {})".format(self.block_ref.block_uuid,
                                                                            self.resource_savings, ops_list)
        return ret

    def init_container(self):
        self.ops = {}
        total_ops = 0
        num_double_entries = 0
        for bb in self.block_ref.brick_list:
            for boi in bb.ops:
                bo = bb.ops[boi]
                on = bo.op_call
                total_ops += 1
                if on not in self.ops.keys():
                    self.ops[on] = bo
                else:
                    if isinstance(self.ops[on], ArchOp):
                        is_compatible = True
                        orig_dims = self.ops[on].dims
                        if len(orig_dims.inp) != len(bo.dims.inp) or orig_dims.inp != bo.dims.inp:
                            is_compatible = False
                        if len(orig_dims.out) != len(bo.dims.out) or orig_dims.out != bo.dims.out:
                            is_compatible = False
                        if len(orig_dims.param) != len(bo.dims.param) or orig_dims.param != bo.dims.param:
                            is_compatible = False

                        if (not is_compatible) and (not self.block_ref.selected_osg.supports_op_padding):
                            orig_op = self.ops[on]
                            new_e = [orig_op, bo]
                            self.ops[on] = new_e
                            num_double_entries += 1
                        # TODO: check for padding options?
                    else:
                        # is already a double entry
                        self.ops[on].append(bo)
                        num_double_entries += 1

        engine_ops = len(self.ops) + num_double_entries
        self.resource_savings = total_ops/engine_ops

    def build(self, build_tool):
        self.block_ref.build_tool = build_tool
        assert self.block_ref.selected_osg != placeholderOSG
        assert len(self.block_ref.brick_list) == len(self.block_ref.selected_contracts)
        self.block_ref.selected_osg.build_container(self, build_tool, self.block_ref.selected_contracts)


