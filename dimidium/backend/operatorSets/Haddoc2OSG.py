#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        DOSA OSG to implement haddoc2 on FPGAs
#  *
#  *

from dimidium.backend.operatorSets.BaseOSG import BaseOSG
from dimidium.backend.devices.dosa_device import DosaHwClasses
from dimidium.middleend.archGen.ArchBrick import ArchBrick
from dimidium.backend.buildTools.BaseBuild import BaseHwBuild
from dimidium.backend.operatorSets.relay_ops import op as relay_op_list


class Haddoc2OSG(BaseOSG):

    def __init__(self):
        super().__init__('haddoc2 OSG', [DosaHwClasses.FPGA_generic, DosaHwClasses.FPGA_xilinx], '/t/b/a',
                         BaseHwBuild('fpga_dummy'))
        self.relay2osg = {}

    def init(self, dosa_hw_classes_dict):
        self.select_dosa_hw_types(dosa_hw_classes_dict)

    def generate_brick(self, brick_node: ArchBrick):
        pass

    def generate_bricks(self, brick_nodes: [ArchBrick]):
        # to generate subsequent bricks at once
        pass

    def comm_wrap_brick(self, todo):
        pass

    def estimate_flops_brick(self, brick_node: ArchBrick):
        pass

