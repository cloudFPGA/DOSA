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
from dimidium.backend.codeGen.VhdlEntity import VhdlEntity


class BaseBuild(metaclass=abc.ABCMeta):

    def __init__(self, name, build_dir=None):
        self.name = name
        self.build_dir = build_dir
        self.node_folder_name = None
        # self.out_dir = out_dir # TODO
        self.global_Makefile = None
        self.makefile_targets = {}

    def create_build_dir(self, node_id: int):
        self.node_folder_name = 'node_{}'.format(node_id)
        new_dir_path = os.path.abspath("{}/{}/".format(dosa_singleton.config.global_build_dir, self.node_folder_name))
        os.system("mkdir -p {}".format(new_dir_path))
        self.build_dir = new_dir_path

    def get_node_folder_name(self):
        return self.node_folder_name

    @abc.abstractmethod
    def write_build_scripts(self):
        print("[DOSA:Build:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def add_makefile_entry(self, path, target):
        print("[DOSA:Build:ERROR] NOT YET IMPLEMENTED.")


class BaseHwBuild(BaseBuild, metaclass=abc.ABCMeta):

    def __init__(self, name, target_device, build_dir=None):
        super().__init__(name, build_dir)
        self.global_vhdl_entity_path = None
        self.global_vhdl_dir = None
        self.global_tcl = None
        self.ip_dirs_list = {}
        self.target_device = target_device
        self.tcl_lines = []

    @abc.abstractmethod
    def add_ip_dir(self, block_id, path=None):
        print("[DOSA:Build:ERROR] NOT YET IMPLEMENTED.")

    def add_tcl_entry(self, tcl_line):
        self.tcl_lines.append(tcl_line)


class BaseSwBuild(BaseBuild, metaclass=abc.ABCMeta):

    def __init__(self, name, target_device=None, build_dir=None):
        super().__init__(name, build_dir)
        self.global_cpp = None
        self.global_hpp = None
        self.lib_dirs = {}
        self.target_deive = target_device

    @abc.abstractmethod
    def add_lib_dir(self, arch_block, path=None):
        print("[DOSA:Build:ERROR] NOT YET IMPLEMENTED.")


class HwBuildTopVhdl(BaseHwBuild, metaclass=abc.ABCMeta):

    def __init__(self, name, target_device, build_dir=None):
        super().__init__(name, target_device, build_dir)
        self.topVhdl = VhdlEntity()


