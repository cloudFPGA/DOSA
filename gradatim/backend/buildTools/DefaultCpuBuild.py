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
#  *    =============================================
#  *     Created: Nov 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Build class for CPU nodes (default, ZRLMPI)
#  *
#  *

from gradatim.backend.buildTools.BaseBuild import BaseSwBuild


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

