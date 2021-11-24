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

from dimidium.lib.util import BrickImplTypes
from dimidium.middleend.archGen.ArchBlock import ArchBlock


class EngineContainer(object):

    def __init__(self, arch_block):
        assert arch_block.block_impl_type == BrickImplTypes.ENGINE
        self.block_ref = arch_block
        self.ops = {}
        self.resource_savings = 0
        self.build_container()

    def __repr__(self):
        ops_list = [k for k in self.ops.keys()]
        ret = "EngineContainer(block_ref: {}, savings: {}, ops: {})".format(self.block_ref.block_uuid,
                                                                            self.resource_savings, ops_list)
        return ret

    def build_container(self):
        self.ops = {}
        total_ops = 0
        for bb in self.block_ref.brick_list:
            for boi in bb.ops:
                bo = bb.ops[boi]
                on = bo.op_call
                total_ops += 1
                # TODO: check also for kernel size etc.
                if on not in self.ops.keys():
                    self.ops[on] = bo
        engine_ops = len(self.ops)
        self.resource_savings = total_ops/engine_ops

