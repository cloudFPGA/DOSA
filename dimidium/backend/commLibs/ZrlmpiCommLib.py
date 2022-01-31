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
