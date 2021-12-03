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

import os
import abc
import dimidium.lib.singleton as dosa_singleton


class BaseBuild(metaclass=abc.ABCMeta):

    def __init__(self, name, build_dir=None, out_dir=None):
        self.name = name
        self.build_dir = build_dir
        self.out_dir = out_dir
        self.global_Makefile = None
        self.makefile_targets = {}

    def create_build_dir(self, node_id: int):
        new_dir_path = os.path.abspath("{}/node_{}/".format(dosa_singleton.config.global_build_dir, node_id))
        os.system("mkdir -p {}".format(new_dir_path))
        self.build_dir = new_dir_path

    @abc.abstractmethod
    def create_global_Makefile(self):
        print("[DOSA:Build:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def add_makefile_entry(self, path, target):
        print("[DOSA:Build:ERROR] NOT YET IMPLEMENTED.")


class BaseHwBuild(BaseBuild, metaclass=abc.ABCMeta):

    def __init__(self, name, target_device, build_dir=None, out_dir=None):
        super().__init__(name, build_dir, out_dir)
        self.global_vhdl = None
        self.global_vhdl_dir = None
        self.global_tcl = None
        self.ip_dirs = {}
        self.target_device = target_device

    @abc.abstractmethod
    def add_ip_dir(self, arch_block, path=None):
        print("[DOSA:Build:ERROR] NOT YET IMPLEMENTED.")


class BaseSwBuild(BaseBuild, metaclass=abc.ABCMeta):

    def __init__(self, name, target_device=None, build_dir=None, out_dir=None):
        super().__init__(name, build_dir, out_dir)
        self.global_cpp = None
        self.global_hpp = None
        self.lib_dirs = {}
        self.target_deive = target_device

    @abc.abstractmethod
    def add_lib_dir(self, arch_block, path=None):
        print("[DOSA:Build:ERROR] NOT YET IMPLEMENTED.")

