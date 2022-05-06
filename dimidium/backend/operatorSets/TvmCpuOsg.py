#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        DOSA OSG to execute TVM operations on CPU
#  *
#  *

from dimidium.lib.util import deep_update, BrickImplTypes
from dimidium.backend.operatorSets.relay_ops import op as relay_op_list
from dimidium.backend.operatorSets.BaseOSG import BaseOSG
from dimidium.backend.devices.dosa_device import DosaHwClasses
from dimidium.middleend.archGen.ArchBrick import ArchBrick
from dimidium.middleend.archGen.OperationContract import OperationContract
from dimidium.lib.dosa_dtype import complete_dtype_list


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
        placeholder_contr = OperationContract(op, target_hw, self, impl_type, float('inf'), 1.0, 1.0,
                                              "not-yet-implemented", 0.0, 0.0)
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

