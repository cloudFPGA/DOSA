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

from dimidium.lib.util import deep_update
from dimidium.backend.operatorSets.relay_ops import op as relay_op_list
from dimidium.backend.operatorSets.BaseOSG import BaseOSG
from dimidium.backend.devices.dosa_device import DosaHwClasses
from dimidium.backend.devices.builtin import vCPU_x86
from dimidium.middleend.archGen.ArchBrick import ArchBrick
from dimidium.backend.buildTools.BaseBuild import BaseSwBuild


class TvmCpuOsg(BaseOSG):

    def __init__(self):
        super().__init__('Tvm CPU OSG', [DosaHwClasses.CPU_generic, DosaHwClasses.CPU_x86], '/t/b/a',
                         BaseSwBuild('tvm_dummy'))

    def init(self, dosa_hw_classes_dict):
        self.select_dosa_hw_types(dosa_hw_classes_dict)
        # this one will support everything
        # self.relay2osg = {x: self._generate_tvm_module for x in relay_op_list}
        self.relay2osg = deep_update(self.relay2osg, True)

    def generate_brick(self, brick_node: ArchBrick):
        pass

    def generate_bricks(self, brick_nodes: [ArchBrick]):
        # to generate subsequent bricks at once
        pass

    # def _generate_tvm_module(self, tvm_handle):
    #     return

    def comm_wrap_brick(self, todo):
        pass

    def estimate_flops_brick(self, brick_node: ArchBrick):
        pass

