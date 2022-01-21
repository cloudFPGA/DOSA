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

import os

from dimidium.backend.buildTools.BaseBuild import BaseHwBuild
from dimidium.backend.devices.dosa_device import DosaHwClasses, DosaBaseHw


class cFBuild1(BaseHwBuild):

    def __init__(self, name, target_device: DosaBaseHw, build_dir=None, out_dir=None):
        super().__init__(name, target_device, build_dir, out_dir)
        self.global_vhdl = None
        self.global_vhdl_dir = None
        self.global_tcl = None
        self.basic_structure_created = False
        self.global_hls_dir = None

    def _create_basic_structure(self):
        os.system("mkdir -p {0}/ROLE/hdl {0}/ROLE/hls {0}/ROLE/tcl".format(self.build_dir))
        me_abs_dir = os.path.dirname(os.path.realpath(__file__))
        my_templates = os.path.abspath(me_abs_dir + '/templates/cFBuild1/')
        os.system("cp {0}/Makefile {1}/ROLE/Makefile".format(my_templates, self.build_dir))
        os.system("cp {0}/Role.vhdl {1}/ROLE/hdl/Role.vhdl".format(my_templates, self.build_dir))
        os.system("cp {0}/tcl/* {1}/ROLE/tcl/".format(my_templates, self.build_dir))
        self.global_vhdl = "{}/ROLE/hdl/Role.vhdl".format(self.build_dir)
        self.global_vhdl_dir = "{}/ROLE/hdl/".format(self.build_dir)
        self.global_hls_dir = "{}/ROLE/hls/".format(self.build_dir)
        self.basic_structure_created = True

    def add_ip_dir(self, arch_block, path=None, vhdl_only=False, hybrid=False):
        if not self.basic_structure_created:
            self._create_basic_structure()
        new_id = arch_block.block_uuid
        ip_dir_list = []
        hls_ip_dir = True
        vhdl_ip_dir = False
        if vhdl_only:
            hls_ip_dir = False
            vhdl_ip_dir = True
        if hybrid:
            hls_ip_dir = True
            vhdl_ip_dir = True
        if vhdl_ip_dir:
            new_path = "{}/block_{}".format(self.global_vhdl_dir, arch_block.block_uuid)
            os.system("mkdir -p {}".format(new_path))
            ip_dir_list.append(new_path)
        if hls_ip_dir:
            new_path = "{}/block_{}".format(self.global_hls_dir, arch_block.block_uuid)
            os.system("mkdir -p {}".format(new_path))
            ip_dir_list.append(new_path)
        arch_block.ip_dir = ip_dir_list
        self.ip_dirs_list[new_id] = ip_dir_list
        if len(ip_dir_list) == 1:
            return ip_dir_list[0]
        return ip_dir_list

    def create_global_Makefile(self):
        # TODO: better part of buildscripts?
        print("[DOSA:Build:ERROR] NOT YET IMPLEMENTED.")

    def add_makefile_entry(self, path, target):
        if path not in self.makefile_targets.keys():
            self.makefile_targets[path] = target

    def write_buildscripts(self):
        # TODO: write tcl lines to copied template file
        #  write global Makefile
        print("[DOSA:Build:ERROR] NOT YET IMPLEMENTED.")


