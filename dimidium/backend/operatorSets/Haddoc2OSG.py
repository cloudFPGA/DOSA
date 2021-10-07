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
from dimidium.backend.devices import DosaHwClasses, cF_FMKU60_Themisto_1
from dimidium.middleend.archGen.ArchBrick import ArchBrick
from dimidium.backend.buildTools.BaseBuild import BaseHwBuild


class Haddoc2OSG(BaseOSG):

    def __init__(self):
        super().__init__('haddoc2 OSG', DosaHwClasses.FPGA_xilinx, '/t/b/a', BaseHwBuild('fpga_dummy'))
        self.dosaHwTypes = [cF_FMKU60_Themisto_1]
        self.relay2osg = {}

    def annotate_brick(self, brick_node: ArchBrick):
        pass

    def generate_brick(self, brick_node: ArchBrick):
        pass

    def generate_bricks(self, brick_nodes: [ArchBrick]):
        # to generate subsequent bricks at once
        pass

    def comm_wrap_brick(self, todo):
        pass

    def estimate_flops_brick(self, brick_node: ArchBrick):
        pass

