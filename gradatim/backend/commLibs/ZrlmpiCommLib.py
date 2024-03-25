#  /*******************************************************************************
#   * Copyright 2019 -- 2024 IBM Corporation
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
#  *     Created: Jan 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        DOSA communication library for CPUs & FPGAs, based on ZRLMPI
#  *
#  *
import os

from gradatim.backend.buildTools.cFBuild1 import cFBuild1
from gradatim.backend.codeGen.WrapperInterfaces import InterfaceAxisFifo
from gradatim.backend.codeGen.ZrlmpiSwApp import ZrlmpiSwApp
from gradatim.backend.codeGen.ZrlmpiWrapper import ZrlmpiWrapper
from gradatim.backend.codeGen.ZrlmpiSwMultiNodeApp import ZrlmpiSwMultiNodeApp
from gradatim.backend.commLibs.BaseCommLib import BaseCommLib
from gradatim.backend.devices.dosa_device import DosaHwClasses
from gradatim.backend.buildTools.BaseBuild import HwBuildTopVhdl, BaseSwBuild


class ZrlmpiCommLib(BaseCommLib):

    def __init__(self):
        super().__init__('ZRLMPI lib', [DosaHwClasses.FPGA_generic, DosaHwClasses.FPGA_xilinx,
                                        DosaHwClasses.CPU_generic, DosaHwClasses.CPU_x86])
        self.priority = 99

    def build(self, comm_plan, build_tool):
        if isinstance(build_tool, HwBuildTopVhdl):
            used_hls_dir_path = build_tool.add_ip_dir('comm')

            comm_wrapper = ZrlmpiWrapper(comm_plan.node.node_id, used_hls_dir_path, comm_plan)
            comm_wrapper.generate()

            build_tool.add_makefile_entry(used_hls_dir_path, 'all')
            # add also mpe2 to makefile target
            mpe2_dir = os.path.abspath(os.path.join(used_hls_dir_path, '../mpe2/'))
            build_tool.add_makefile_entry(mpe2_dir, 'all')

            zrlmpi_tcl = comm_wrapper.get_tcl_lines_wrapper_inst()
            build_tool.add_tcl_entry(zrlmpi_tcl)
            zrlmpi_vhdl_decl = comm_wrapper.get_wrapper_vhdl_decl_lines()
            zrlmpi_inst_tmpl = comm_wrapper.get_vhdl_inst_tmpl()
            build_tool.topVhdl.set_network_adapter(zrlmpi_vhdl_decl, zrlmpi_inst_tmpl, [InterfaceAxisFifo])
            tcl_tmp, decl_tmp, inst_tmp = comm_wrapper.get_debug_lines()
            build_tool.topVhdl.debug_core.add_new_probes(tcl_tmp, decl_tmp, inst_tmp)
            if isinstance(build_tool, cFBuild1):
                add_constr_lines = comm_wrapper.get_add_constr_lines()
                if add_constr_lines is not None:
                    build_tool.add_additional_constraint_lines(add_constr_lines)
        elif isinstance(build_tool, BaseSwBuild):
            if comm_plan.node.node_id == 0:
                # SW app
                node_dir = build_tool.build_dir
                if len(comm_plan.node.ranks) == 1:
                    comm_inst = ZrlmpiSwApp(comm_plan.node.node_id, node_dir, comm_plan)
                else:
                    comm_inst = ZrlmpiSwMultiNodeApp(comm_plan.node.node_id, node_dir, comm_plan)
                comm_inst.generate()
                # add nothing to SwBuild Tool, since this is the root app (?)
            else:
                print("[DOSA:CommLibBuild:ERROR] not yet implemented.")
        else:
            print("[DOSA:CommLibBuild:ERROR] ZRLMPI for build type {} is not supported.".format(type(build_tool)))
        return 0


