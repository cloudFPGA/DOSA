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
import json

import dimidium.lib.singleton as dosa_singleton
from dimidium.backend.buildTools.BaseBuild import BaseHwBuild, HwBuildTopVhdl
from dimidium.backend.devices.dosa_device import DosaHwClasses, DosaBaseHw
from dimidium.backend.buildTools.lib.cFBuild1.cFCreate.templates.gen_env import __cfenv_small_name__, \
    __cfenv_req_packages__, get_sys_python_env
from dimidium.backend.buildTools.lib.cFBuild1.cFCreate.lib.cf_create import create_cfp_dir_structure, \
    __to_be_defined_key__, copy_templates_and_set_env
from dimidium.backend.buildTools.lib.cFBuild1.cFCreate.templates.cf_sratool import __cfp_json_name__, \
    __sra_dict_template__, __sra_key__

__dosa_config_env_dcps__ = 'DOSA_cFBuild1_used_dcps_path'


class cFBuild1(HwBuildTopVhdl):

    build_wide_structure_created = False

    def __init__(self, name, target_device: DosaBaseHw, build_dir=None):
        super().__init__(name, target_device, build_dir, use_debug=True)
        self.basic_structure_created = False
        self.global_hls_dir = None
        self.constr_lines = []

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
        self.my_makefile = '{}/ROLE/Makefile'.format(self.build_dir)
        # os.system("cp {0}/Role.vhdl {1}/ROLE/hdl/Role.vhdl".format(my_templates, self.build_dir))
        self.topVhdl.set_template('{}/Role.vhdl'.format(my_templates))
        # cFDK setup etc.
        self.my_global_build_lib_dir = os.path.abspath('{}/cFBuild1/'.format(dosa_singleton.config.global_build_dir))
        my_libs = os.path.abspath(me_abs_dir + '/lib/cFBuild1/')
        self.my_libs = my_libs
        global_venv_dir = '{}/{}'.format(self.my_global_build_lib_dir, __cfenv_small_name__)
        self.my_global_venv = global_venv_dir
        self.my_global_dcps = '{}/global_dcps/'.format(self.my_global_build_lib_dir)
        if not self.build_wide_structure_created:
            self._create_build_wide_structure()
        create_cfp_dir_structure(self.build_dir)
        # link cFDK
        os.system('cd {}; ln -s {}/cFDK cFDK'.format(self.build_dir, self.my_global_build_lib_dir))
        # copy templates that should be copied only during create
        os.system("cp {0}/cFCreate/templates/gitignore.template {1}/.gitignore".format(my_libs, self.build_dir))
        os.system("cp {0}/cFCreate/templates/cfdk_Makefile {1}/Makefile".format(my_libs, self.build_dir))
        cf_envs = {'cf_sra': self.target_device.cf_sra, 'cf_mod': self.target_device.cf_mod_type,
                   'roleName1': self.name, 'usedRoleDir': '', 'roleName2': __to_be_defined_key__,
                   'usedRoleDir2': __to_be_defined_key__}
        copy_templates_and_set_env(self.build_dir, cf_envs, False)
        # link dcps
        os.system('cd {}; mkdir -p dcps'.format(self.build_dir))
        # use relative symlinks
        dcp_dir = os.path.abspath(self.build_dir + '/dcps/')
        gdcps_rel = os.path.relpath(self.my_global_dcps, dcp_dir)
        os.system('cd {build}; ln -s {gdcps}/3_top{mod}_STATIC.dcp 3_top{mod}_STATIC.dcp'.
                  format(build=dcp_dir, mod=self.target_device.cf_mod_type, gdcps=gdcps_rel))
        os.system('cd {build}; ln -s {gdcps}/3_top{mod}_STATIC.json 3_top{mod}_STATIC.json'.
                  format(build=dcp_dir, mod=self.target_device.cf_mod_type, gdcps=gdcps_rel))
        # add role and set active
        cfp_json_file = os.path.abspath(self.build_dir + '/' + __cfp_json_name__)
        with open(cfp_json_file, 'r') as json_file:
            cFp_data = json.load(json_file)
        cFp_data[__sra_key__] = __sra_dict_template__
        cFp_data[__sra_key__]['roles'] = [{'name': self.name, 'path': ''}]
        cFp_data[__sra_key__]['active_role'] = self.name
        with open(cfp_json_file, 'w') as json_file:
            json.dump(cFp_data, json_file, indent=4)
        # link venv
        os.system('cd {}; ln -s {} env/{}'.format(self.build_dir, self.my_global_venv, __cfenv_small_name__))
        self.basic_structure_created = True

    def _create_build_wide_structure(self):
        me_abs_dir = os.path.dirname(os.path.realpath(__file__))
        self.my_global_build_lib_dir = os.path.abspath('{}/cFBuild1/'.format(dosa_singleton.config.global_build_dir))
        my_libs = os.path.abspath(me_abs_dir + '/lib/cFBuild1/')
        self.my_libs = my_libs
        my_templates = os.path.abspath(me_abs_dir + '/templates/cFBuild1/')
        os.system('mkdir -p {}/cFDK'.format(self.my_global_build_lib_dir))
        os.system('cp -R {}/cFDK/* {}/cFDK/'.format(my_libs, self.my_global_build_lib_dir))
        self.my_global_dcps = '{}/global_dcps/'.format(self.my_global_build_lib_dir)
        os.system('mkdir -p {}'.format(self.my_global_dcps))
        if __dosa_config_env_dcps__ not in os.environ:
            print("[DOSA:cFBuild1:ERROR] Can't locate 3_static dcp to be used...build will most likely fail. Trying "
                  "to continue.\n")
        else:
            used_dcps_dir = os.path.abspath(os.environ[__dosa_config_env_dcps__])
            os.system('cp {}/3_* {}/'.format(used_dcps_dir, self.my_global_dcps))
        global_venv_dir = '{}/{}'.format(self.my_global_build_lib_dir, __cfenv_small_name__)
        self.my_global_venv = global_venv_dir
        os.system('mkdir -p {}'.format(global_venv_dir))
        cfenv_dir = os.path.abspath(global_venv_dir)
        sys_py_bin = get_sys_python_env()
        # print("[INFO] the python virutalenv for this project on this machine is missing, installing it...")
        os.system('cd {}; virtualenv -p {} {}'
                  .format(os.path.abspath(cfenv_dir + '/../'), sys_py_bin, __cfenv_small_name__))
        os.system('/bin/bash -c "source {}/bin/activate; pip install {}"'.format(cfenv_dir, __cfenv_req_packages__))
        # copy deploy script
        os.system('cp {}/dosa_deploy.py {}/'.format(my_templates, dosa_singleton.config.global_build_dir))
        cFBuild1.build_wide_structure_created = True

    def add_ip_dir(self, block_id, path=None, vhdl_only=False, hybrid=False):
        if not self.basic_structure_created:
            self._create_basic_structure()
        new_id = block_id
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
            new_path = "{}/block_{}".format(self.global_vhdl_dir, new_id)
            os.system("mkdir -p {}".format(new_path))
            ip_dir_list.append(new_path)
        if hls_ip_dir:
            new_path = "{}/block_{}".format(self.global_hls_dir, new_id)
            os.system("mkdir -p {}".format(new_path))
            ip_dir_list.append(new_path)
        # arch_block.ip_dir = ip_dir_list
        self.ip_dirs_list[new_id] = ip_dir_list
        if len(ip_dir_list) == 1:
            return ip_dir_list[0]
        return ip_dir_list

    def add_makefile_entry(self, path, target):
        relpath = os.path.relpath(path, os.path.dirname(self.my_makefile))
        if relpath not in self.makefile_targets.keys():
            self.makefile_targets[relpath] = target

    def _write_tcl_file(self):
        with open('{}/tcl/create_ip_cores.tcl'.format(self.my_templates), 'r') as in_file, \
                open('{}/ROLE/tcl/create_ip_cores.tcl'.format(self.build_dir), 'w') as out_file:
            for line in in_file.readlines():
                if 'DOSA_ADD_tcl_decls' in line:
                    outline = ''
                    for e in self.tcl_lines:
                        if isinstance(e, str):
                            outline += e
                        elif isinstance(e, list):
                            for ee in e:
                                outline += ee
                        outline += '\n'
                else:
                    outline = line
                out_file.write(outline)

    def _write_makefile(self):
        with open('{}/Makefile'.format(self.my_templates), 'r') as in_file, \
                open(self.my_makefile, 'w') as out_file:
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
        # 3. write additional constraints, if existing
        self._write_constr_file()
        #  last, write global Makefile
        self._write_makefile()

    def add_additional_constraint_lines(self, constr_lines):
        self.constr_lines.append(constr_lines)

    def _write_constr_file(self):
        if len(self.constr_lines) > 0:
            os.system('mkdir -p {}/ROLE/xdc/'.format(self.build_dir))
            constr_file = '# automatically DOSA generated additional constraints for the Role\n\n'
            for e in self.constr_lines:
                constr_file += e
                constr_file += '\n'
            constr_file += '\n'
            with open('{}/ROLE/xdc/additional_role_constraints.tcl'.format(self.build_dir), 'w') as out_file:
                out_file.write(constr_file)

