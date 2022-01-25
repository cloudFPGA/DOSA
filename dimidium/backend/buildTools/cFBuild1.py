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

from dimidium.backend.buildTools.BaseBuild import BaseHwBuild, HwBuildTopVhdl
from dimidium.backend.devices.dosa_device import DosaHwClasses, DosaBaseHw


class cFBuild1(HwBuildTopVhdl):

    def __init__(self, name, target_device: DosaBaseHw, build_dir=None, out_dir=None):
        super().__init__(name, target_device, build_dir, out_dir)
        self.basic_structure_created = False
        self.global_hls_dir = None

    def _create_basic_structure(self):
        os.system("mkdir -p {0}/ROLE/hdl {0}/ROLE/hls {0}/ROLE/tcl".format(self.build_dir))
        me_abs_dir = os.path.dirname(os.path.realpath(__file__))
        my_templates = os.path.abspath(me_abs_dir + '/templates/cFBuild1/')
        self.my_templates = my_templates
        # os.system("cp {0}/Makefile {1}/ROLE/Makefile".format(my_templates, self.build_dir))
        os.system("cp {0}/tcl/* {1}/ROLE/tcl/".format(my_templates, self.build_dir))
        os.system("cp -R {0}/hls/triangle_app {1}/ROLE/hls/".format(my_templates, self.build_dir))
        self.global_vhdl_entity_path = "{}/ROLE/hdl/Role.vhdl".format(self.build_dir)
        self.global_vhdl_dir = "{}/ROLE/hdl/".format(self.build_dir)
        self.global_hls_dir = "{}/ROLE/hls/".format(self.build_dir)
        # os.system("cp {0}/Role.vhdl {1}/ROLE/hdl/Role.vhdl".format(my_templates, self.build_dir))
        self.topVhdl.set_template('{}/Role.vhdl'.format(my_templates))
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

    def add_makefile_entry(self, path, target):
        if path not in self.makefile_targets.keys():
            self.makefile_targets[path] = target

    def _write_tcl_file(self):
        with open('{}/tcl/create_ip_cores.tcl'.format(self.my_templates), 'r') as in_file, \
                open('{}/ROLE/tcl/create_ip_cores.tcl'.format(self.build_dir), 'w') as out_file:
            for line in in_file.readlines():
                if 'DOSA_ADD_tcl_decls' in line:
                    outline = ''
                    for e in self.tcl_lines:
                        outline += e
                        outline += '\n'
                else:
                    outline = line
                out_file.write(outline)

    def _write_makefile(self):
        with open('{}/Makefile'.format(self.my_templates), 'r') as in_file, \
                open('{}/ROLE/Makefile'.format(self.build_dir), 'w') as out_file:
            for line in in_file.readlines():
                if 'DOSA_add_make_targets' in line:
                    outline = ''
                    for tp in self.makefile_targets.keys():
                        mt = self.makefile_targets[tp]
                        outline += '\t$(MAKE) -C {tp} {mt}\n'.format(tp=tp, mt=mt)
                else:
                    outline = line
                out_file.write(outline)

    def write_build_scripts(self):
        # 1. write vhdl
        self.topVhdl.write_file(self.global_vhdl_entity_path, self.target_device)
        self.add_tcl_entry(self.topVhdl.get_add_tcl_lines())
        # 2. write tcl lines
        self._write_tcl_file()
        #  write global Makefile
        self._write_makefile()


