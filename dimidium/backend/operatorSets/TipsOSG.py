#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: May 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        DOSA OSG to implement TIPS Engines on FPGAs
#  *
#  *
import os
import numpy as np
import tvm
import tvm.relay as relay
import math
import json

from dimidium.backend.buildTools.BaseBuild import HwBuildTopVhdl
from dimidium.backend.buildTools.cFBuild1 import cFBuild1
from dimidium.backend.operatorSets.BaseOSG import BaseOSG
from dimidium.backend.devices.dosa_device import DosaHwClasses
from dimidium.lib.dosa_dtype import get_bitwidth_of_DosaDtype, DosaDtype, complete_dtype_list
from dimidium.lib.util import BrickImplTypes
from dimidium.middleend.archGen.ArchBrick import ArchBrick
from dimidium.backend.operatorSets.relay_ops import op as relay_op_list
from dimidium.backend.codeGen.WrapperInterfaces import InterfaceAxisFifo, wrapper_default_interface_bitwidth
from dimidium.middleend.archGen.OperationContract import OperationContract
from dimidium.backend.operatorSets.lib.util import get_avg_util_dict_bytes_based, get_share_of_FPGA_resources
import dimidium.lib.units as units

__filedir__ = os.path.dirname(os.path.abspath(__file__))
__db_path__ = __filedir__ + '/osg_impl_db.json'


class TipsOSG(BaseOSG):

    def __init__(self):
        super().__init__('Tips OSG', [DosaHwClasses.FPGA_xilinx], complete_dtype_list,
                         [BrickImplTypes.ENGINE])
        self.priority = 99
        me_abs_dir = os.path.dirname(os.path.realpath(__file__))
        self.my_template_dir = os.path.abspath(me_abs_dir + '/lib/tips/')
        self.util_db = {}
        self.avg_util_dict = {}
        self.pipeline_tensor_store = 1

    def _init_util_db_(self):
        with open(__db_path__, 'r') as infile:
            util_data = json.load(infile)
        my_util = util_data[self.name]
        self.util_db = {}
        compl_list = []
        for e in my_util:
            if e['device'] not in self.util_db:
                self.util_db[e['device']] = [e]
            else:
                self.util_db[e['device']].append(e)
            compl_list.append(e)
        self.avg_util_dict = get_avg_util_dict_bytes_based(compl_list, consider_paramB=True)

    def _get_impl_prediction(self, op_str, inpB, paramB, device, consider_paramB=False, custom_byte_factor=1.0):
        relevant_entries = []
        exact_matches = []
        # TODO: prefer entries with shorter ops list?
        for dk in self.util_db:
            if dk == device.type_str:
                for e in self.util_db[dk]:
                    if op_str in e['ops']:
                        relevant_entries.append(e)
                        if e['latency_lim_per_tensor_cycl'] > 0:
                            if consider_paramB:
                                if e['inpB'] == inpB and e['paramB'] == paramB:
                                    exact_matches.append(e)
                            else:
                                if e['inpB'] == inpB:
                                    exact_matches.append(e)
        res_dict = {}
        used_fallback = False
        if len(relevant_entries) == 0:
            res_dict = self.avg_util_dict
            used_fallback = True
        elif len(exact_matches) > 0:
            res_dict = get_avg_util_dict_bytes_based(exact_matches, consider_paramB=consider_paramB)
        else:
            res_dict = get_avg_util_dict_bytes_based(relevant_entries, consider_paramB=consider_paramB)
        ret_dict = {}
        bytes_total = inpB
        if consider_paramB:
            bytes_total += paramB
        bytes_total *= custom_byte_factor
        ret_dict['LUTLOG'] = res_dict['LUTLOG'] * bytes_total
        ret_dict['LUTMEM'] = res_dict['LUTMEM'] * bytes_total
        ret_dict['Registers'] = res_dict['Registers'] * bytes_total
        ret_dict['BRAM'] = res_dict['BRAM'] * bytes_total
        ret_dict['DSPs'] = res_dict['DSPs'] * bytes_total
        ret_dict['latency_lim_per_tensor_cycl'] = res_dict['latency_lim_per_tensor_cycl'] * (inpB + paramB)
        wrapper_dict = {'LUTLOG': 0.0, 'LUTMEM': 0.0, 'Registers': 0.0, 'BRAM': 0.0, 'DSPs': 0.0}
        return ret_dict, wrapper_dict, used_fallback

    def init(self, dosa_hw_classes_dict, priority_internal):
        self.priority_internal = priority_internal
        self.select_dosa_hw_types(dosa_hw_classes_dict)
        self._init_util_db_()
        # relay2osg annotation,
        #  based on https://github.com/DreamIP/haddoc2/blob/master/lib/python/parseNetTopology.py
        for e in self.relay2osg['nn']:
            if 'conv2d' in e:
                self.relay2osg['nn'][e] = self._param_parse_conv, self._predict_conv
            elif 'bias_add' in e:
                self.relay2osg['nn'][e] = self._generate_hdl_biasAdd, self._predict_bias
            elif 'pool2d' in e:
                self.relay2osg['nn'][e] = self._param_parse_pool, self._predict_pool
            elif 'tanh' in e:
                self.relay2osg['nn'][e] = self._generate_hdl_tanh_instance, self._predict_tanh
            elif 'relu' in e:
                self.relay2osg['nn'][e] = self._generate_hdl_relu, self._predict_relu
            elif 'flatten' in e:
                self.relay2osg['nn'][e] = self._generate_hdl_flatten_instance, self._predict_flatten_drop
            elif 'dropout' in e:
                self.relay2osg['nn'][e] = self._generate_hdl_dropout_instance, self._predict_flatten_drop
        for e in self.relay2osg:
            if type(e) == dict:
                continue
            elif 'tanh' in e:
                self.relay2osg[e] = self._generate_hdl_tanh_instance, self._predict_tanh

    def build_block(self, arch_block, build_tool, selected_contracts):
        pass

    def build_container(self, container, build_tool, selected_contracts):
        pass

