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

from dimidium.backend.operatorSets.BaseOSG import BaseOSG
from dimidium.backend.devices import DosaHwClasses, vCPU_x86
from dimidium.middleend.archGen.ArchBrick import ArchBrick
from dimidium.backend.buildTools.BaseBuild import BaseSwBuild


class TvmCpuOsg(BaseOSG):

    def __init__(self):
        super().__init__('Tvm CPU OSG', DosaHwClasses.CPU_x86, '/t/b/a', BaseSwBuild('tvm_dummy'))
        self.dosaHwTypes = [vCPU_x86]

    def annotate_brick(self, brick_node: ArchBrick):
        pass

    def generate_brick(self, brick_node: ArchBrick):
        pass

    def comm_wrap_brick(self, todo):
        pass

    def estimate_flops_brick(self, brick_node: ArchBrick):
        pass

