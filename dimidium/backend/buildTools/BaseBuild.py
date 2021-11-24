#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Base classes for DOSA build tools
#  *
#  *

import abc


class BaseBuild(metaclass=abc.ABCMeta):

    def __init__(self, name, build_dir, out_dir):
        self.name = name
        self.build_dir = build_dir
        self.out_dir = out_dir
        self.global_Makefile = None


class BaseHwBuild(BaseBuild, metaclass=abc.ABCMeta):

    def __init__(self, name, build_dir, out_dir):
        super().__init__(name, build_dir, out_dir)
        self.global_vhdl = None
        self.global_vhdl_dir = None
        self.global_tcl = None
        self.ip_dirs = {}

    @abc.abstractmethod
    def add_ip_dir(self, arch_block, path=None):
        print("[DOSA:Build:ERROR] NOT YET IMPLEMENTED.")


class BaseSwBuild(BaseBuild, metaclass=abc.ABCMeta):

    def __init__(self, name, build_dir, out_dir):
        super().__init__(name, build_dir, out_dir)
        self.global_cpp = None
        self.global_hpp = None
        self.lib_dirs = {}

    @abc.abstractmethod
    def add_lib_dir(self, arch_block, path=None):
        print("[DOSA:Build:ERROR] NOT YET IMPLEMENTED.")

