#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        DOSA OSG to implement haddoc2 on FPGAs
#  *
#  *

import os

from dimidium.backend.buildTools.BaseBuild import BaseHwBuild
from dimidium.backend.buildTools.cFBuild1 import cFBuild1
from dimidium.backend.operatorSets.BaseOSG import BaseOSG
from dimidium.backend.devices.dosa_device import DosaHwClasses
from dimidium.lib.util import BrickImplTypes
from dimidium.middleend.archGen.ArchBrick import ArchBrick
from dimidium.backend.operatorSets.relay_ops import op as relay_op_list


class Haddoc2OSG(BaseOSG):

    def __init__(self):
        super().__init__('haddoc2 OSG', [DosaHwClasses.FPGA_generic, DosaHwClasses.FPGA_xilinx], '/t/b/a',
                         [BrickImplTypes.STREAM])
        self.priority = 99
        me_abs_dir = os.path.dirname(os.path.realpath(__file__))
        self.my_hdl_template_folder = os.path.abspath(me_abs_dir + '/../3rd_party_libs/haddoc2/lib/hdl/')

    def init(self, dosa_hw_classes_dict, priority_internal):
        self.priority_internal = priority_internal
        self.select_dosa_hw_types(dosa_hw_classes_dict)
        # relay2osg annotation,
        #  based on https://github.com/DreamIP/haddoc2/blob/master/lib/python/parseNetTopology.py
        for e in self.relay2osg['nn']:
            if 'conv1d' in e or 'conv2d' in e:
                self.relay2osg['nn'][e] = self._generate_hdl_conv_instance
            elif 'pool1d' in e or 'pool2d' in e:
                self.relay2osg['nn'][e] = self._generate_hdl_pool_instance
            elif 'tanh' in e:
                self.relay2osg['nn'][e] = self._generate_hdl_tanh_instance

    def _copy_hdl_lib(self, target_hdl_dir):
        os.system("cp -n {}/* {}/".format(self.my_hdl_template_folder, target_hdl_dir))

    def build_block(self, arch_block, build_tool):
        assert isinstance(build_tool, BaseHwBuild)
        if isinstance(build_tool, cFBuild1):
            # for cF, everything under hdl/ will be included
            used_dir_path = build_tool.add_ip_dir(arch_block, is_vhdl=True)
        else:
            used_dir_path = build_tool.add_ip_dir(arch_block)
        # first, copy all lib files; use global_vhdl_dir to have it only once per node
        self._copy_hdl_lib(build_tool.global_vhdl_dir)

    def build_container(self, container, build_tool):
        pass

    # def generate_brick(self, brick_node: ArchBrick):
    #     pass

    # def generate_bricks(self, brick_nodes: [ArchBrick]):
    #     # to generate subsequent bricks at once
    #     pass

    # def comm_wrap_brick(self, todo):
    #     pass

    def estimate_flops_brick(self, brick_node: ArchBrick):
        pass

    def _generate_hdl_conv_instance(self, todo):
        return

    def _generate_hdl_pool_instance(self, todo):
        return

    def _generate_hdl_tanh_instance(self, todo):
        # is merged with ConvLayer.vhd?
        # but separate TanH exist, so could be added
        return


