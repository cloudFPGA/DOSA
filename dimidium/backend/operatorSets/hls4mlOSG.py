#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        DOSA OSG to implement hls4ml on FPGAs
#  *
#  *

from dimidium.backend.operatorSets.BaseOSG import BaseOSG
from dimidium.backend.devices import DosaHwClasses, cF_FMKU60_Themisto_1
from dimidium.middleend.archGen.ArchBrick import ArchBrick
from dimidium.backend.buildTools.BaseBuild import BaseHwBuild


class Hls4mlOSG(BaseOSG):

    def __init__(self):
        super().__init__('hls4ml OSG', DosaHwClasses.FPGA_xilinx, '/t/b/a', BaseHwBuild('fpga_dummy'))

    def annotate_brick(self, brick_node: ArchBrick):
        pass

    def generate_brick(self, brick_node: ArchBrick):
        pass

    def comm_wrap_brick(self, todo):
        pass

    def estimate_flops_brick(self, brick_node: ArchBrick):
        pass

