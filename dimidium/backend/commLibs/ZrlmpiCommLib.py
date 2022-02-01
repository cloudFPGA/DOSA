#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Jan 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        DOSA communication library for CPUs & FPGAs, based on ZRLMPI
#  *
#  *
import os

from dimidium.backend.codeGen.WrapperInterfaces import InterfaceAxisFifo
from dimidium.backend.codeGen.ZrlmpiWrapper import ZrlmpiWrapper
from dimidium.backend.commLibs.BaseCommLib import BaseCommLib
from dimidium.backend.devices.dosa_device import DosaHwClasses
from dimidium.backend.buildTools.BaseBuild import HwBuildTopVhdl


class ZrlmpiCommLib(BaseCommLib):

    def __init__(self):
        super().__init__('ZRLMPI lib', [DosaHwClasses.FPGA_generic, DosaHwClasses.FPGA_xilinx,
                                        DosaHwClasses.CPU_generic, DosaHwClasses.CPU_x86])
        self.priority = 99

    def build(self, comm_plan, build_tool):
        assert isinstance(build_tool, HwBuildTopVhdl)
        used_hls_dir_path = build_tool.add_ip_dir('comm')

        comm_wrapper = ZrlmpiWrapper(comm_plan.node.node_id, used_hls_dir_path, comm_plan)
        comm_wrapper.generate()

        build_tool.add_makefile_entry(used_hls_dir_path, 'all')
        # add also mpe2 to makefile target
        mpe2_dir = os.path.join(used_hls_dir_path, '../mpe2/')
        build_tool.add_makefile_entry(mpe2_dir, 'all')

        zrlmpi_tcl = comm_wrapper.get_tcl_lines_wrapper_inst()
        build_tool.add_tcl_entry(zrlmpi_tcl)
        zrlmpi_vhdl_decl = comm_wrapper.get_wrapper_vhdl_decl_lines()
        zrlmpi_inst_tmpl = comm_wrapper.get_vhdl_inst_tmpl()
        build_tool.topVhdl.set_network_adapter(zrlmpi_vhdl_decl, zrlmpi_inst_tmpl, [InterfaceAxisFifo])
        return 0

