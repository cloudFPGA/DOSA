#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Nov 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Build class for CPU nodes (default, ZRLMPI)
#  *
#  *

from dimidium.backend.buildTools.BaseBuild import BaseSwBuild


class DefaultCpuBuild(BaseSwBuild):

    def __init__(self, name, build_dir=None, out_dir=None):
        super().__init__(name, build_dir, out_dir)
        self.global_cpp = None
        self.global_hpp = None

    def add_lib_dir(self, arch_block, path=None):
        pass

    def write_build_scripts(self):
        # print("[DOSA:Build:ERROR] NOT YET IMPLEMENTED.")
        print("[DOSA:DefaultCpuBuild:WARNING] Nothing to do for write_build_scripts.")

    def add_makefile_entry(self, path, target):
        # print("[DOSA:Build:ERROR] NOT YET IMPLEMENTED.")
        print("[DOSA:DefaultCpuBuild:WARNING] Nothing to do for add_makefile_entry.")

