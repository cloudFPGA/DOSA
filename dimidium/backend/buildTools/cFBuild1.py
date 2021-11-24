#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Nov 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Build class for cF nodes (Themisto)
#  *
#  *

from dimidium.backend.buildTools.BaseBuild import BaseHwBuild


class cFBuild1(BaseHwBuild):

    def __init__(self, name, build_dir, out_dir):
        super().__init__(name, build_dir, out_dir)
        self.global_vhdl = None
        self.global_vhdl_dir = None
        self.global_tcl = None

    def add_ip_dir(self, arch_block, path=None):
        pass