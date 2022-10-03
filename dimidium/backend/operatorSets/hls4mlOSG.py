#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        DOSA OSG to implement hls4ml on FPGAs
#  *
#  *

import os
import math
import numpy as np
import json
import tvm.relay

import dimidium.lib.singleton as dosa_singleton
from dimidium.backend.buildTools.BaseBuild import BaseHwBuild, HwBuildTopVhdl
from dimidium.backend.codeGen.Hls4mlWrapper import Hls4mlWrapper
from dimidium.backend.codeGen.WrapperInterfaces import InterfaceAxisFifo, wrapper_default_interface_bitwidth
from dimidium.backend.operatorSets.BaseOSG import BaseOSG
from dimidium.backend.devices.dosa_device import DosaHwClasses
from dimidium.backend.operatorSets.lib.util import get_avg_util_dict_bytes_based, get_share_of_FPGA_resources
from dimidium.lib import units
from dimidium.middleend.archGen.ArchBrick import ArchBrick
from dimidium.lib.util import BrickImplTypes
from dimidium.backend.operatorSets.relay_ops import op as relay_op_list
from dimidium.backend.operatorSets.osgUtils import convert_IntImm_array
from dimidium.lib.dosa_dtype import get_bitwidth_of_DosaDtype, DosaDtype, DosaDtype_is_signed, DosaDtype_to_string, \
    complete_dtype_list
from dimidium.backend.operatorSets.lib.hls4ml.dosa_to_hls import dosa_to_hls
from dimidium.backend.operatorSets.lib.hls4ml.DosaFileReader import OsgDataReader
from dimidium.backend.operatorSets.lib.hls4ml.dosa_to_hls import dosa_to_hls
from dimidium.middleend.archGen.OperationContract import OperationContract

__filedir__ = os.path.dirname(os.path.abspath(__file__))
__db_path__ = __filedir__ + '/osg_impl_db.json'


def _get_next_unrolling_factor(paral_grade):
    u_paral_grade = max(1.0, paral_grade)
    possible_unrolling_factors = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    possible_unrolling_factors.reverse()
    for e in possible_unrolling_factors:
        if u_paral_grade >= e:
            return int(e)
    return int(1.0)


def get_loop_unrolling_factors(req_parallelization_grade):
    if req_parallelization_grade < 1.0:
        req_parallelization_grade = 1.0
    if req_parallelization_grade > 128.0:
        req_parallelization_grade = 128.0  # TODO
    remaining_paral_grade = req_parallelization_grade
    innermost = _get_next_unrolling_factor(remaining_paral_grade)
    remaining_paral_grade -= innermost
    inner = _get_next_unrolling_factor(remaining_paral_grade)
    remaining_paral_grade -= inner
    outer = _get_next_unrolling_factor(remaining_paral_grade)
    remaining_paral_grade -= outer
    outermost = _get_next_unrolling_factor(remaining_paral_grade)
    return outermost, outer, inner, innermost


class Hls4mlOSG(BaseOSG):

    def __init__(self):
        super().__init__('hls4ml OSG', [DosaHwClasses.FPGA_xilinx], complete_dtype_list,
                         [BrickImplTypes.STREAM])
        # no DosaHwClasses.FPGA_generic, since it is bound to xilinx?
        self.priority = 92
        self.existing_layer_names = []
        # self.suggested_max_block_length = 1
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

    def _get_impl_prediction(self, op, target_hw, impl_type, consider_paramB=False, fallback_ops=None,
                             consider_outB=False, custom_byte_factor=1.0, custom_latency=None, max_param_dim=-1):
        # if impl_type != BrickImplTypes.STREAM or \
        #         (target_hw.hw_class != DosaHwClasses.FPGA_xilinx and target_hw.hw_class != DosaHwClasses.FPGA_generic):
        #     return None
        if max_param_dim > 0:
            op_param_dim = 1
            for d in op.dims.param:
                op_param_dim *= d
            if op_param_dim > max_param_dim:
                print("[DOSA:hls4ml:INFO] Can't offer an implementation for {}, due to exceeded parameter size."
                      .format(repr(op)))
                return None
        relevant_entries = []
        # TODO: prefer entries with shorter ops list?
        fallback_entries = []
        exact_matches = []
        op_str = op.op_call.split('.')[-1]
        dtype_str = 'int8'  # default?
        if op.used_dtype != DosaDtype.UNKNOWN:
            dtype_str = repr(op.used_dtype)
        if dosa_singleton.uc['overwrite_fixed_point_dtypes']:
            dtype_str = 'Q{}'.format(get_bitwidth_of_DosaDtype(op.used_dtype))
        for dk in self.util_db:
            if dk == target_hw.type_str:
                for e in self.util_db[dk]:
                    if op_str in e['ops'] and dtype_str in e['dtype']:
                        relevant_entries.append(e)
                        if op.input_bytes == e['inpB'] and e['latency_lim_per_tensor_cycl'] > 0:
                            if (consider_paramB and op.parameter_bytes == e['paramB']) or (not consider_paramB):
                                exact_matches.append(e)
                    if fallback_ops is not None:
                        for f in fallback_ops:
                            if f in e['ops']:
                                fallback_entries.append(e)
                                break
        res_dict = {}
        used_fallback = False
        if len(relevant_entries) == 0 and len(fallback_entries) == 0:
            res_dict = self.avg_util_dict
            used_fallback = True
        elif len(exact_matches) > 0:
            res_dict = get_avg_util_dict_bytes_based(exact_matches, consider_paramB=consider_paramB,
                                                     consider_ops_num=False, consider_outB=consider_outB)
        else:
            if len(relevant_entries) == 0:
                relevant_entries = fallback_entries
                used_fallback = True
            res_dict = get_avg_util_dict_bytes_based(relevant_entries, consider_paramB=consider_paramB,
                                                     consider_ops_num=False, consider_outB=consider_outB)
        if consider_paramB:
            bytes_total = op.parameter_bytes
        else:
            bytes_total = op.input_bytes
        if consider_outB:
            bytes_total += op.output_bytes
        bytes_total *= custom_byte_factor
        util_dict = {}
        util_dict['LUTLOG'] = res_dict['LUTLOG'] * bytes_total
        util_dict['LUTMEM'] = res_dict['LUTMEM'] * bytes_total
        util_dict['Registers'] = res_dict['Registers'] * bytes_total
        util_dict['BRAM'] = res_dict['BRAM'] * bytes_total
        util_dict['DSPs'] = res_dict['DSPs'] * bytes_total
        util_dict['latency_lim_per_tensor_cycl'] = res_dict['latency_lim_per_tensor_cycl'] * op.input_bytes
        wrapper_dict = {}
        wrapper_dict['LUTLOG'] = res_dict['wrapper']['LUTLOG'] * bytes_total
        wrapper_dict['LUTMEM'] = res_dict['wrapper']['LUTMEM'] * bytes_total
        wrapper_dict['Registers'] = res_dict['wrapper']['Registers'] * bytes_total
        wrapper_dict['BRAM'] = res_dict['wrapper']['BRAM'] * bytes_total
        wrapper_dict['DSPs'] = res_dict['wrapper']['DSPs'] * bytes_total

        fpga_utility = target_hw.get_resource_dict()['FPGA_utility']
        proc_share = get_share_of_FPGA_resources(fpga_utility, util_dict)
        wrapper_share = get_share_of_FPGA_resources(fpga_utility, wrapper_dict)
        # proc_comp_share = (proc_share['LUTLOG'] + proc_share['DSPs']) / 2
        proc_comp_share = proc_share['LUTLOG']  # we know we hardly use DSPs..
        proc_mem_share = (proc_share['LUTMEM'] + proc_share['Registers'] + proc_share['BRAM']) / 3
        # wrapper_comp_share = (wrapper_share['LUTLOG'] + wrapper_share['DSPs']) / 2
        wrapper_comp_share = wrapper_share['LUTLOG']  # we know we hardly use DSPs...
        wrapper_mem_share = (wrapper_share['LUTMEM'] + wrapper_share['Registers'] + wrapper_share['BRAM']) / 3
        if custom_latency is None:
            latency_ns = util_dict['latency_lim_per_tensor_cycl'] * target_hw.get_performance_dict()['fpga_clk_ns']
            iter_hz = 1 / (latency_ns * units.nanoU)
        else:
            if custom_latency <= 0:
                custom_latency = 1
            latency_ns = custom_latency * target_hw.get_performance_dict()['fpga_clk_ns']
            iter_hz = 1 / (latency_ns * units.nanoU)
        offer = OperationContract(op, target_hw, self, BrickImplTypes.STREAM, iter_hz, proc_comp_share, proc_mem_share,
                                  'default', wrapper_comp_share, wrapper_mem_share, proc_share, wrapper_share)
        offer_list = [offer]
        if not used_fallback:
            proc_share['LUTLOG'] *= 0.5
            proc_share['LUTMEM'] *= 0.5
            proc_share['Registers'] *= 0.5
            proc_share['BRAM'] *= 0.5
            proc_share['DSPs'] *= 0.5
            # util_dict['latency_lim_per_tensor_cycl'] *= 2
            # wrapper stays
            offer_05 = OperationContract(op, target_hw, self, BrickImplTypes.STREAM, iter_hz * 0.5,
                                         proc_comp_share * 0.5,
                                         proc_mem_share * 0.5, 'conf:mult_limit=0.5', wrapper_comp_share,
                                         wrapper_mem_share,
                                         proc_share, wrapper_share)
            # TODO
            # offer_list.append(offer_05)
        # TODO: add alternative offers
        #  e.g. with alternative reuse factor?
        return offer_list

    def init(self, dosa_hw_classes_dict, priority_internal):
        self.priority_internal = priority_internal
        self.select_dosa_hw_types(dosa_hw_classes_dict)
        self._init_util_db_()
        # relay2osg annotation,
        #  based on https://github.com/fastmachinelearning/hls4ml/blob/master/hls4ml/model/hls_layers.py
        #  and https://github.com/fastmachinelearning/hls4ml/tree/master/hls4ml/converters/
        for e in self.relay2osg['nn']:
            if 'conv1d' in e:
                self.relay2osg['nn'][e] = self._generate_hls_conv1d, \
                                          lambda op, thw, it: self._get_impl_prediction(op, thw, it,
                                                                                        consider_paramB=True,
                                                                                        consider_outB=True,
                                                                                        fallback_ops=['conv2d',
                                                                                                      'dense'],
                                                                                        custom_byte_factor=1.8,
                                                                                        max_param_dim=400)
            elif 'conv2d' in e:
                self.relay2osg['nn'][e] = self._generate_hls_conv2d, \
                                          lambda op, thw, it: self._get_impl_prediction(op, thw, it,
                                                                                        consider_paramB=True,
                                                                                        consider_outB=True,
                                                                                        fallback_ops=['conv1d',
                                                                                                      'dense'],
                                                                                        custom_byte_factor=1.8,
                                                                                        max_param_dim=400)
            elif 'global' in e and 'pool1d' in e:
                self.relay2osg['nn'][e] = self._generate_hls_globalPool1d, \
                                          lambda op, thw, it: self._get_impl_prediction(op, thw, it,
                                                                                        consider_paramB=False,
                                                                                        fallback_ops=['pool2d',
                                                                                                      'pool1d', 'add'])
            elif 'global' in e and 'pool2d' in e:
                self.relay2osg['nn'][e] = self._generate_hls_globalPool2d, \
                                          lambda op, thw, it: self._get_impl_prediction(op, thw, it,
                                                                                        consider_paramB=False,
                                                                                        fallback_ops=['pool2d',
                                                                                                      'pool1d', 'add'])
            elif 'pool1d' in e:
                self.relay2osg['nn'][e] = self._generate_hls_pool1d, \
                                          lambda op, thw, it: self._get_impl_prediction(op, thw, it,
                                                                                        consider_paramB=False,
                                                                                        fallback_ops=['pool2d',
                                                                                                      'pool1d', 'add'])
            elif 'pool2d' in e:
                self.relay2osg['nn'][e] = self._generate_hls_pool2d, \
                                          lambda op, thw, it: self._get_impl_prediction(op, thw, it,
                                                                                        consider_paramB=False,
                                                                                        fallback_ops=['pool2d',
                                                                                                      'pool1d', 'add'])
            elif 'prelu' in e:
                self.relay2osg['nn'][e] = self._generatae_hls_prelu, \
                                          lambda op, thw, it: self._get_impl_prediction(op, thw, it,
                                                                                        consider_paramB=False,
                                                                                        fallback_ops=['relu',
                                                                                                      'softmax'])
            elif 'relu' in e:
                self.relay2osg['nn'][e] = self._generate_hls_parAct, \
                                          lambda op, thw, it: self._get_impl_prediction(op, thw, it,
                                                                                        consider_paramB=False,
                                                                                        fallback_ops=['prelu',
                                                                                                      'softmax'],
                                                                                        custom_latency=int(op.dims.inp[-1]/2))
            elif 'softmax' in e:
                self.relay2osg['nn'][e] = self._generate_hls_softmax, \
                                          lambda op, thw, it: self._get_impl_prediction(op, thw, it,
                                                                                        consider_paramB=False,
                                                                                        fallback_ops=['prelu', 'relu'],
                                                                                        custom_latency=int(op.dims.inp[-1]/2))
            elif 'dense' in e:
                self.relay2osg['nn'][e] = self._generate_hls_dense, \
                                          lambda op, thw, it: self._get_impl_prediction(op, thw, it,
                                                                                        consider_paramB=True,
                                                                                        consider_outB=True,
                                                                                        custom_byte_factor=1.2,
                                                                                        fallback_ops=['conv1d',
                                                                                                      'conv2d'])
            elif 'batch_norm' in e:
                self.relay2osg['nn'][e] = self._generate_hls_batchNorm, \
                                          lambda op, thw, it: self._get_impl_prediction(op, thw, it,
                                                                                        consider_paramB=True,
                                                                                        fallback_ops=None)
            elif 'pad' in e:
                self.relay2osg['nn'][e] = self._generate_hls_padding, \
                                          lambda op, thw, it: OperationContract(op, thw, self, it, BaseOSG._pseudo_infinity_, 0.0,
                                                                                0.0, 'dummy op', 0.0, 0.0)
            elif 'bias_add' in e:
                self.relay2osg['nn'][e] = self._generate_hls_biasAdd, \
                                          lambda op, thw, it: self._get_impl_prediction(op, thw, it,
                                                                                        consider_paramB=False,
                                                                                        fallback_ops=None,
                                                                                        custom_latency=int(op.dims.inp[-1]/2))
            elif 'flatten' in e:
                self.relay2osg['nn'][e] = self._generate_hls_flatten, \
                                          lambda op, thw, it: OperationContract(op, thw, self, it, BaseOSG._pseudo_infinity_, 0.0,
                                                                                0.0, 'dummy op', 0.0, 0.0)
            elif 'dropout' in e:
                self.relay2osg['nn'][e] = self._generate_hls_dropout, \
                                          lambda op, thw, it: OperationContract(op, thw, self, it, BaseOSG._pseudo_infinity_, 0.0,
                                                                                0.0, 'dummy op', 0.0, 0.0)
        for e in self.relay2osg:
            if type(self.relay2osg[e]) == dict:
                continue
            if ('tan' in e or 'sin' in e or 'cos' in e) and 'is' not in e:
                self.relay2osg[e] = self._generate_hls_act, \
                                    lambda op, thw, it: self._get_impl_prediction(op, thw, it,
                                                                                  consider_paramB=True,
                                                                                  fallback_ops=['tan', 'sin', 'cos'],
                                                                                  custom_latency=int(op.dims.inp[-1]/2))
            elif 'add' in e or 'sub' in e or 'mul' in e or 'avg' in e \
                 or 'max' in e or 'min' in e or 'concat' in e or 'sum' in e:
                self.relay2osg[e] = self._generate_hls_merge, \
                                    lambda op, thw, it: self._get_impl_prediction(op, thw, it,
                                                                                  consider_paramB=False,
                                                                                  fallback_ops=['add', 'sub', 'mul',
                                                                                                'avg', 'max', 'min',
                                                                                                'concat', 'sum'],
                                                                                  custom_latency=int(op.dims.inp[-1]/2))
            elif 'transpose' in e:
                self.relay2osg[e] = self._generate_hls_transpose, \
                                    lambda op, thw, it: OperationContract(op, thw, self, it, BaseOSG._pseudo_infinity_, 0.0,
                                                                          0.0, 'dummy op', 0.0, 0.0)
                                    # TODO: is transpose really for free in hls4ml?
            elif 'reshape' in e or 'expand_dims' in e or 'squeeze' in e:
                self.relay2osg[e] = self._generate_hls_reshape, \
                                    lambda op, thw, it: OperationContract(op, thw, self, it, BaseOSG._pseudo_infinity_, 0.0,
                                                                          0.0, 'dummy op', 0.0, 0.0)
        # not covered hls4ml classes:
        #  GarNet, Resize, SeparableConv2D, DepthwiseConv2D

    def _create_unique_layer_name(self, op_name):
        base_str = op_name.replace('.', '_')
        name_cnt = 1
        while base_str in self.existing_layer_names:
            base_str += "_{}".format(name_cnt)
            name_cnt += 1
        self.existing_layer_names.append(base_str)
        return base_str

    def write_makefile(self, ip_dir, project_name, reset=False, csim=True, synth=True,
                       cosim=False, validation=False, export=True, vsynth=False):
        echo_cmd = os.popen('which echo').read().rstrip()
        bc_cmd = os.popen('which bc').read().rstrip()
        target_file = '{}/Makefile'.format(ip_dir)
        with open(target_file, 'w') as m_file:
            m_file.write('\n# DOSA: Makefile automatically generated by hls4ml OSG\n\n')
            new_str = ('export dosa_hls4ml_reset={reset}\nexport dosa_hls4ml_csim={csim}' +
                       '\nexport dosa_hls4ml_synth={synth}\nexport dosa_hls4ml_cosim={cosim}' +
                       '\nexport dosa_hls4ml_validation={validation}\nexport dosa_hls4ml_export={export}' +
                       '\nexport dosa_hls4ml_vsynth={vsynth}\n\n') \
                .format(reset=int(reset), csim=int(csim), synth=int(synth), cosim=int(cosim),
                        validation=int(validation), export=int(export), vsynth=int(vsynth))
            m_file.write(new_str)
            m_file.write('all: {}_prj/solution1/impl/ip\n\n'.format(project_name))
            new_str = ('{project}_prj/solution1/impl/ip:\n\t@date +%s > .tmp_stamp_3\n\tvivado_hls -f build_prj.tcl \n'
                       + '\t@cat {project}_prj/solution1/syn/report/{project}_csynth.rpt\n\t@echo "-" > .tmp_stamp_2' +
                       '\n\t@date +%s > .tmp_stamp_1') \
                .format(project=project_name)
            new_str += '\n\t@cp {project}_prj/solution1/syn/report/{project}_csynth.rpt {rpt_dir}/{project}.rpt' \
                .format(project=project_name, rpt_dir=dosa_singleton.config.global_report_dir)
            new_str += ('\n\t@{echo} -n "HLS4ML build time: "' +
                        "\n\t@/bin/bash -c \"cat <(cat .tmp_stamp_* | tr '\\n' ' ') <(echo '') | {bc} -l\"") \
                .format(echo=echo_cmd, bc=bc_cmd)
            m_file.write(new_str)
            m_file.write('\n\n')

    def _get_num_of_non_zero_weights(self, data: np.ndarray):
        cnt = 0
        for x in np.nditer(data):
            if -1e-20 < x < 1e20:
                cnt += 1
        return cnt

    def get_max_num_of_mult(self, data_dict):
        max = 0
        for vn in data_dict['variables']:
            if isinstance(data_dict['variables'][vn], np.ndarray):
                nm = self._get_num_of_non_zero_weights(data_dict['variables'][vn])
            else:
                if isinstance(data_dict['variables'][vn]['node'], tvm.relay.expr.Call):
                    continue
                nm = data_dict['variables'][vn]['node'].type_annotation.shape[0]
                for e in data_dict['variables'][vn]['node'].type_annotation.shape[1::]:
                    nm *= e
            if nm > max:
                max = nm
        return int(max)

    def build_block(self, arch_block, build_tool, selected_contracts):
        assert isinstance(build_tool, HwBuildTopVhdl)
        assert len(selected_contracts) == len(arch_block.brick_list)
        used_dir_path = build_tool.add_ip_dir(arch_block.block_uuid)
        project_name = 'ArchBlock_{}'.format(arch_block.block_uuid)
        used_dtype = DosaDtype.int32
        cur_w = 0
        for bb in arch_block.brick_list:
            cur_dt = bb.used_dtype
            bitw = get_bitwidth_of_DosaDtype(cur_dt)
            if bitw > cur_w:
                used_dtype = cur_dt
                cur_w = bitw
        precision_string = ''
        accum_string = ''
        if used_dtype == DosaDtype.float16 or used_dtype == DosaDtype.float32:
            precision_string = 'ap_fixed<16,6>'  # TODO
            accum_string = precision_string
        else:
            int_bits = cur_w
            if not dosa_singleton.uc['overwrite_fixed_point_dtypes']:
                precision_string = 'ap_uint<{}>'.format(cur_w)
            else:
                fractional_bits = dosa_singleton.uc['overwrite_dtypes']['fixed_point_fraction_bits']
                int_bits = cur_w - fractional_bits
                # according to xlinix documentation
                #  https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Overview-of-Arbitrary-Precision-Fixed-Point-Data-Types
                #  https://github.com/Xilinx/HLS_arbitrary_Precision_Types/blob/200a9aecaadf471592558540dc5a88256cbf880f/include/ap_fixed_base.h#L809
                # ap_fixed<8,2> means 6 fractional bits, and the sign bit is included in the 2 integer bits
                # so no extra subtraction
                # if DosaDtype_is_signed(used_dtype):
                #     int_bits -= 1
                precision_string = 'ap_fixed<{},{}, AP_RND_CONV, AP_SAT_SYM>'.format(cur_w, int_bits)
            if dosa_singleton.uc['use_extra_accum_dtype']:
                accum_factor = dosa_singleton.uc['overwrite_dtypes']['accum_bits_factor']
                accum_string = 'ap_fixed<{},{}, AP_RND_CONV, AP_SAT_SYM>'.format(cur_w * accum_factor,
                                                                                 int_bits * accum_factor)
            else:
                accum_string = precision_string

        first_used_dtype = arch_block.brick_list[0].used_dtype
        # input_batch_shape = arch_block.brick_list[0].ops[0].dims.inp
        if len(arch_block.brick_list[0].ops[0].dims.inp) == 4:
            # swap to channels_last format
            input_batch_shape = [0, 0, 0, 0]
            input_batch_shape[0] = arch_block.brick_list[0].ops[0].dims.inp[0]
            input_batch_shape[1] = arch_block.brick_list[0].ops[0].dims.inp[2]
            input_batch_shape[2] = arch_block.brick_list[0].ops[0].dims.inp[3]
            input_batch_shape[3] = arch_block.brick_list[0].ops[0].dims.inp[1]
        else:
            input_batch_shape = arch_block.brick_list[0].ops[0].dims.inp
        if input_batch_shape[0] != 1:
            print("[DOSA:OSG:ERROR] hls4ml only supports models with batch_size 1")
            exit(-1)

        # reuse_factor_stream = 1
        # reuse_factor_stream = 4  # TODO
        reuse_factor_stream = 32  # works so far...TODO
        if np.prod(input_batch_shape) < reuse_factor_stream:
            # reuse_factor_stream = math.floor(np.prod(input_batch_shape)/2) * 2
            reuse_factor_stream = 1  # i.e. deactivating? # TODO
        reuse_factor_engine = 2
        precision_dict = {'default': precision_string,
                          'accum': accum_string}
        hls_config = {'Model': {
            # 'Precision': precision_string,
            'Precision': precision_dict,
            'ReuseFactor': reuse_factor_engine,
            'Strategy': 'Resource'}}

        # TODO: tune hls pragmas...
        if arch_block.block_impl_type == BrickImplTypes.STREAM:
            hls_config['Model']['Strategy'] = 'Latency'
            hls_config['Model']['ReuseFactor'] = reuse_factor_stream

        hls_model_config = {'OutputDir': used_dir_path, 'ProjectName': project_name, 'Backend': 'Vivado',
                            'XilinxPart': build_tool.target_device.part_string, 'Board': None,
                            'ClockPeriod': build_tool.target_device.clock_period_ns,
                            'IOType': 'io_stream',  # the interface is then even more weired... (or does not compile)
                            # 'IOType': 'io_parallel',  # TODO
                            # 'IOType': 'io_serial',  # is deprecated from version 0.5.0, but is the only one working
                            'HLSConfig': hls_config}  # ,
        # 'KerasJson': 'KERAS_3layer.json', 'KerasH5': 'KERAS_3layer_weights.h5'}  # TODO

        model_arch2 = {'backend': 'tensorflow',
                       'class_name': 'Model',
                       'config': {'input_layers': [['input_1', 0, 0]], 'layers': [{'class_name': 'InputLayer',
                                                                                   'config': {
                                                                                       'batch_input_shape': [None, 16],
                                                                                       'dtype': 'float32',
                                                                                       'name': 'input_1',
                                                                                       'sparse': False},
                                                                                   'inbound_nodes': [],
                                                                                   'name': 'input_1'},
                                                                                  {'class_name': 'Dense',
                                                                                   'config': {'activation': 'relu',
                                                                                              'activity_regularizer': None,
                                                                                              'bias_constraint': None,
                                                                                              'bias_initializer':
                                                                                                  {
                                                                                                      'class_name': 'VarianceScaling',
                                                                                                      'config': {
                                                                                                          'distribution': 'uniform',
                                                                                                          'mode': 'fan_in',
                                                                                                          'scale': 1.0,
                                                                                                          'seed': None}},
                                                                                              'bias_regularizer': None,
                                                                                              'kernel_constraint': None,
                                                                                              'kernel_initializer': {
                                                                                                  'class_name': 'VarianceScaling',
                                                                                                  'config': {
                                                                                                      'distribution': 'uniform',
                                                                                                      'mode': 'fan_in',
                                                                                                      'scale': 1.0,
                                                                                                      'seed': None}},
                                                                                              'kernel_regularizer': {
                                                                                                  'class_name': 'L1L2',
                                                                                                  'config': {'l1': 0.0,
                                                                                                             'l2': 0.0}},
                                                                                              'name': 'fc1_relu',
                                                                                              'trainable': True,
                                                                                              'units': 64,
                                                                                              'use_bias': True},
                                                                                   'inbound_nodes': [
                                                                                       [['input_1', 0, 0, {}]]],
                                                                                   'name': 'fc1_relu'},
                                                                                  {'class_name': 'Dense',
                                                                                   'config': {'activation': 'relu',
                                                                                              'activity_regularizer': None,
                                                                                              'bias_constraint': None,
                                                                                              'bias_initializer': {
                                                                                                  'class_name': 'VarianceScaling',
                                                                                                  'config': {
                                                                                                      'distribution': 'uniform',
                                                                                                      'mode': 'fan_in',
                                                                                                      'scale': 1.0,
                                                                                                      'seed': None}},
                                                                                              'bias_regularizer': None,
                                                                                              'kernel_constraint': None,
                                                                                              'kernel_initializer': {
                                                                                                  'class_name': 'VarianceScaling',
                                                                                                  'config': {
                                                                                                      'distribution': 'uniform',
                                                                                                      'mode': 'fan_in',
                                                                                                      'scale': 1.0,
                                                                                                      'seed': None}},
                                                                                              'kernel_regularizer': {
                                                                                                  'class_name': 'L1L2',
                                                                                                  'config': {'l1': 0.0,
                                                                                                             'l2': 0.0}},
                                                                                              'name': 'fc2_relu',
                                                                                              'trainable': True,
                                                                                              'units': 32,
                                                                                              'use_bias': True},
                                                                                   'inbound_nodes': [
                                                                                       [['fc1_relu', 0, 0, {}]]],
                                                                                   'name': 'fc2_relu'},
                                                                                  {'class_name': 'Dense',
                                                                                   'config': {'activation': 'relu',
                                                                                              'activity_regularizer': None,
                                                                                              'bias_constraint': None,
                                                                                              'bias_initializer': {
                                                                                                  'class_name': 'VarianceScaling',
                                                                                                  'config': {
                                                                                                      'distribution': 'uniform',
                                                                                                      'mode': 'fan_in',
                                                                                                      'scale': 1.0,
                                                                                                      'seed': None}},
                                                                                              'bias_regularizer': None,
                                                                                              'kernel_constraint': None,
                                                                                              'kernel_initializer': {
                                                                                                  'class_name': 'VarianceScaling',
                                                                                                  'config': {
                                                                                                      'distribution': 'uniform',
                                                                                                      'mode': 'fan_in',
                                                                                                      'scale': 1.0,
                                                                                                      'seed': None}},
                                                                                              'kernel_regularizer': {
                                                                                                  'class_name': 'L1L2',
                                                                                                  'config': {'l1': 0.0,
                                                                                                             'l2': 0.0}},
                                                                                              'name': 'fc3_relu',
                                                                                              'trainable': True,
                                                                                              'units': 32,
                                                                                              'use_bias': True},
                                                                                   'inbound_nodes': [
                                                                                       [['fc2_relu', 0, 0, {}]]],
                                                                                   'name': 'fc3_relu'},
                                                                                  {'class_name': 'Dense',
                                                                                   'config': {'activation': 'softmax',
                                                                                              'activity_regularizer': None,
                                                                                              'bias_constraint': None,
                                                                                              'bias_initializer': {
                                                                                                  'class_name': 'VarianceScaling',
                                                                                                  'config': {
                                                                                                      'distribution': 'uniform',
                                                                                                      'mode': 'fan_in',
                                                                                                      'scale': 1.0,
                                                                                                      'seed': None}},
                                                                                              'bias_regularizer': None,
                                                                                              'kernel_constraint': None,
                                                                                              'kernel_initializer': {
                                                                                                  'class_name': 'VarianceScaling',
                                                                                                  'config': {
                                                                                                      'distribution': 'uniform',
                                                                                                      'mode': 'fan_in',
                                                                                                      'scale': 1.0,
                                                                                                      'seed': None}},
                                                                                              'kernel_regularizer': {
                                                                                                  'class_name': 'L1L2',
                                                                                                  'config': {'l1': 0.0,
                                                                                                             'l2': 0.0}},
                                                                                              'name': 'output_softmax',
                                                                                              'trainable': True,
                                                                                              'units': 5,
                                                                                              'use_bias': True},
                                                                                   'inbound_nodes': [
                                                                                       [['fc3_relu', 0, 0, {}]]],
                                                                                   'name': 'output_softmax'}],
                                  'name': 'model_1', 'output_layers': [['output_softmax', 0, 0]]},
                       'keras_version': '2.0.0'}

        reader = OsgDataReader(hls_model_config)
        model_arch = {'backend': 'dosa', 'class_name': 'Model',  # 'Model" to emulate TF >=2.3
                      'config': {'input_layers': [], 'layers': [], 'name': project_name, 'output_layers': []}}

        input_layer = {'class_name': 'InputLayer',
                       'config': {'batch_input_shape': input_batch_shape,
                                  # 'dtype': first_used_dtype.name,
                                  # TODO
                                  'dtype': precision_string,
                                  'name': 'input_1', 'sparse': False,
                                  # 'data_format': 'channels_first'},
                                  'data_format': 'channels_last'},
                       'inbound_nodes': [], 'name': 'input_1'}

        model_arch['config']['input_layers'].append(['input_1', 0, 0])  # TODO, dynamic?
        model_arch['config']['layers'].append(input_layer)

        layer_names_by_op_id = {}
        layer_names_ordered = []
        layer_confs = []
        last_layer_name = ''
        wrapper_first_brick = None
        wrapper_last_brick = None
        wrapper_first_op = None
        wrapper_last_op = None
        contr_i = 0
        for bb in arch_block.brick_list:
            # for op in bb.local_op_iter_gen():
            if wrapper_first_brick is None:
                wrapper_first_brick = bb
            wrapper_last_brick = bb
            bb_speedup_req = bb.req_flops / bb.flops
            skip_i = []
            cur_brick_contr = selected_contracts[contr_i]
            for op_i in bb.ops.keys():
                if op_i in skip_i:
                    continue
                op = bb.ops[op_i]
                op_c = cur_brick_contr.get_contract_to_op(op)
                # contract details
                mult_limit_factor = 1.0
                # if selected_contract.osg_internal_id == 'default'
                if op_c.osg_intern_id[0:5] == 'conf:':
                    conf_str = op_c.osg_intern_id[5:]
                    conf_list = conf_str.split(';')
                    for c in conf_list:
                        if 'mult_limit' in c:
                            mult_limit_factor = float(c.split('=')[1])
                            print("[DOSA:OSG:INFO] Using adapted mult_limit_factor {}.".format(mult_limit_factor))

                if wrapper_first_op is None:
                    wrapper_first_op = op
                wrapper_last_op = op
                next_i = op_i + 1
                next_op = None
                if next_i in bb.ops.keys():
                    next_op = bb.ops[next_i]
                    wrapper_last_op = next_op
                next_next_i = op_i + 2
                next_next_op = None
                if next_next_i in bb.ops.keys():
                    next_next_op = bb.ops[next_next_i]
                    wrapper_last_op = next_next_op
                layer_name = self._create_unique_layer_name(op.name)
                layer_names_by_op_id[op.global_op_id] = layer_name
                layer_names_ordered.append(layer_name)
                osg_func = self._get_osg_func(op.op_call)
                # print(op.op_call)
                conf, data, consumed_opt_ops = osg_func(op, layer_name, next_op, next_next_op)
                # here only stream
                # conf['ReuseFactor'] = reuse_factor_stream
                # conf['mult_limit'] = math.ceil(bb.req_flops / reuse_factor_stream)
                # conf['mult_limit'] = bb.req_flops
                # mult_limit = 1000
                if data is not None:
                    reader.add_data_dict(data)
                    mult_limit = self.get_max_num_of_mult(data)
                else:
                    tmp_dict = {'variables': {}}
                    t_cnt = 0
                    for e in op.tvm_args['by_position']:
                        tmp_dict['variables'][str(t_cnt)] = e
                        t_cnt += 1
                    mult_limit = self.get_max_num_of_mult(tmp_dict)
                conf['mult_limit'] = mult_limit * mult_limit_factor
                exec_simple_s = op.flops * build_tool.target_device.clock_period_s
                op_req_flops = op.flops * bb_speedup_req
                # op_req_latency_s = 1/bb_speedup_req
                if op_req_flops == 0:
                    op_req_latency_s = 1e-6
                else:
                    op_req_latency_s = op.flops / op_req_flops
                req_paral_grade = exec_simple_s / op_req_latency_s
                # outermost, outer, inner, innermost = get_loop_unrolling_factors(req_paral_grade)
                # conf['loop_lim_outermost'] = outermost
                # conf['loop_lim_outer'] = outer
                # conf['loop_lim_inner'] = inner
                # conf['loop_lim_innermost'] = innermost
                if consumed_opt_ops >= 1:
                    skip_i.append(next_i)
                if consumed_opt_ops >= 2:
                    skip_i.append(next_next_i)
                layer_confs.append(conf)
                last_layer_name = layer_name
            contr_i += 1

        model_arch['config']['layers'].extend(layer_confs)
        model_arch['config']['output_layers'].append([last_layer_name, 0, 0])

        print("[DOSA:hls4mlOSG:INFO] starting hls4ml tool...")
        hls_model = dosa_to_hls(hls_model_config, reader, model_arch)
        hls_model.write()  # i.e. write HLS sources
        # hls_model.config.set_source_script('/opt/xilinx/Vivado/2019.2/settings64.sh')
        # synth_entry = {'ip_dir': used_dir_path, 'func': hls_model.build}
        # arch_block.add_synth_entry(synth_entry)
        self.write_makefile(used_dir_path, project_name, reset=True)
        build_tool.add_makefile_entry(used_dir_path, 'all')
        # wrapper & interface generation
        wrapper_input_fifo = InterfaceAxisFifo('input_{}'.format(arch_block.block_uuid),
                                               wrapper_first_brick.input_bw_Bs, build_tool.target_device)
        if build_tool.topVhdl.next_proc_comp_cnt == 0:
            # i.e. we are connected to the input
            wrapper_input_fifo.bitwidth = wrapper_default_interface_bitwidth
        if_in_bitw = wrapper_input_fifo.get_if_bitwidth()
        wrapper_output_fifo = InterfaceAxisFifo('output_{}'.format(arch_block.block_uuid),
                                                wrapper_last_brick.output_bw_Bs, build_tool.target_device)
        if len(arch_block.parent_node.arch_block_list) < 2:
            # we are the only one, so output must also be set
            wrapper_output_fifo.bitwidth = wrapper_default_interface_bitwidth
        if_out_bitw = wrapper_output_fifo.get_if_bitwidth()
        # if_fifo_name = wrapper_input_fifo.get_if_name()
        if_axis_tcl = wrapper_input_fifo.get_tcl_lines()
        build_tool.add_tcl_entry(if_axis_tcl)

        wrapper_dir_path = build_tool.add_ip_dir('{}_wrapper'.format(arch_block.block_uuid))
        block_wrapper = Hls4mlWrapper(arch_block.block_uuid, wrapper_first_op.dims.inp, wrapper_last_op.dims.out,
                                      get_bitwidth_of_DosaDtype(wrapper_first_brick.used_dtype),
                                      get_bitwidth_of_DosaDtype(wrapper_last_brick.used_dtype),
                                      if_in_bitw, if_out_bitw, wrapper_dir_path)
        block_wrapper.generate()
        build_tool.add_makefile_entry(wrapper_dir_path, 'all')
        wrapper_inst_tcl = block_wrapper.get_tcl_lines_wrapper_inst()
        build_tool.add_tcl_entry(wrapper_inst_tcl)
        wrapper_decl = block_wrapper.get_wrapper_vhdl_decl_lines()
        wrapper_inst_tmpl = block_wrapper.get_vhdl_inst_tmpl()

        build_tool.topVhdl.add_proc_comp_inst(arch_block, wrapper_decl, wrapper_inst_tmpl, wrapper_input_fifo,
                                              wrapper_output_fifo)

        # adding debug
        tcl_tmp, decl_tmp, inst_tmp = wrapper_input_fifo.get_debug_lines()
        build_tool.topVhdl.debug_core.add_new_probes(tcl_tmp, decl_tmp, inst_tmp)
        # unsure if output will be used --> add debug lines later
        # tcl_tmp, decl_tmp, inst_tmp = wrapper_output_fifo.get_debug_lines()
        # build_tool.topVhdl.debug_core.add_new_probes(tcl_tmp, decl_tmp, inst_tmp)
        tcl_tmp, decl_tmp, inst_tmp = block_wrapper.get_debug_lines()
        build_tool.topVhdl.debug_core.add_new_probes(tcl_tmp, decl_tmp, inst_tmp)
        return 0

    def build_container(self, container, build_tool, selected_contracts):
        print("[DOSA:Build:ERROR] hls4ml OSG was asked to build an engine container, but it can't. STOP.")
        exit(1)
        return -1

    # def generate_brick(self, brick_node: ArchBrick):
    #     pass

    # def generate_bricks(self, brick_nodes: [ArchBrick]):
    #     # to generate subsequent bricks at once
    #     pass

    # def comm_wrap_brick(self, todo):
    #     pass

    # def estimate_flops_brick(self, brick_node: ArchBrick):
    #     pass

    def _generate_hls_conv1d(self, op, layer_name, next_op=None, next_next_op=None):
        return

    def _generate_hls_conv2d(self, op, layer_name, next_op=None, next_next_op=None):
        ret3 = {'class_name': 'Conv2D',
                'config': {'activation': 'relu', 'activity_regularizer': None, 'batch_input_shape': [None, 8, 8, 1],
                           'bias_constraint': None,
                           'bias_initializer': {'class_name': 'VarianceScaling',
                                                'config': {'distribution': 'uniform', 'mode': 'fan_avg', 'scale': 1.0,
                                                           'seed': None}},
                           'bias_regularizer': None,
                           'data_format': 'channels_last', 'dilation_rate': [1, 1],
                           'dtype': 'float32', 'filters': 2, 'kernel_constraint': None,
                           'kernel_initializer': {'class_name': 'VarianceScaling',
                                                  'config': {'distribution': 'uniform', 'mode': 'fan_avg', 'scale': 1.0,
                                                             'seed': None}},
                           'kernel_regularizer': None, 'kernel_size': [3, 3],
                           'name': 'conv2d_1',
                           'padding': 'same', 'strides': [1, 1], 'trainable': True, 'use_bias': True}}
        ret = {'class_name': 'Conv2D', 'config': {}}
        # 'config' must contain 'data_format' 'channels_first'
        conv_config = {'data_format': 'channels_last', 'dtype': op.used_dtype.name, 'trainable': False}
        assert op.tvm_node.attrs.data_layout == 'NCHW'  # TODO
        conv_config['kernel_constraint'] = None  # TODO?
        conv_config['kernel_initializer'] = None  # TODO?
        conv_config['kernel_regularizer'] = None
        # conv_config['batch_input_shape'] = op.dims.inp
        conv_config['batch_input_shape'] = [0, 0, 0, 0]
        conv_config['batch_input_shape'][0] = op.dims.inp[0]
        conv_config['batch_input_shape'][1] = op.dims.inp[2]
        conv_config['batch_input_shape'][2] = op.dims.inp[3]
        conv_config['batch_input_shape'][3] = op.dims.inp[1]
        # conv_config['kernel_size'] = [op.dims.param[2], op.dims.param[3]]
        conv_config['kernel_size'] = convert_IntImm_array(op.tvm_node.attrs.kernel_size)
        conv_config['name'] = layer_name
        conv_config['padding'] = 'same'  # TODO, op.tvm_node.attrs.padding
        conv_config['strides'] = convert_IntImm_array(op.tvm_node.attrs.strides)
        conv_config['dilation_rate'] = convert_IntImm_array(op.tvm_node.attrs.dilation)
        conv_config['use_bias'] = False
        conv_config['filters'] = 1
        # further: name, class_name, activation, use_bias and 'epsilon'?

        # data must have var names 'kernel' and 'bias' (also 'depthwise_kernel?')
        data = {'layer_name': layer_name, 'variables': {}, 'data_format': {}}
        data['variables']['kernel'] = op.tvm_args['by_position'][1]['ref'].data.numpy()
        data['data_format']['kernel'] = 'channels_first'

        consumed_opt_ops = 0
        if next_op is not None and next_op.op_call == 'nn.bias_add':
            conv_config['use_bias'] = True
            conv_config['bias_constraint'] = None
            conv_config['bias_initializer'] = None
            conv_config['bias_regularizer'] = None
            data['variables']['bias'] = next_op.tvm_args['by_position'][1]['ref'].data.numpy()
            data['data_format']['bias'] = 'channels_first'
            conv_config['filters'] += 1
            consumed_opt_ops += 1
        else:
            conv_config['use_bias'] = False
            conv_config['bias_constraint'] = None
            conv_config['bias_initializer'] = None
            conv_config['bias_regularizer'] = None

        if next_next_op is not None and next_next_op.op_call == 'nn.relu':
            conv_config['activity_regularizer'] = None
            conv_config['activation'] = 'relu'
            consumed_opt_ops += 1
        elif next_next_op is not None and next_next_op.op_call == 'nn.softmax':
            conv_config['activity_regularizer'] = None
            conv_config['activation'] = 'softmax'
            consumed_opt_ops += 1
        elif next_next_op is not None and (next_next_op.op_call == 'nn.tanh' or next_next_op.op_call == 'tanh'):
            conv_config['activity_regularizer'] = None
            conv_config['activation'] = 'tanh'
            consumed_opt_ops += 1
        else:
            conv_config['activity_regularizer'] = None
            # conv_config['activation'] = None  # don't put the key in

        # assemble dict
        ret['config'] = conv_config
        return ret, data, consumed_opt_ops

    def _generate_hls_pool1d(self, op, layer_name, next_op=None, next_next_op=None):
        ret = {'class_name': 'MaxPooling1D', 'config': {}}
        layer_config = {'data_format': 'channels_last', 'dtype': op.used_dtype.name, 'trainable': False}
        layer_config['batch_input_shape'] = [0, 0, 0, 0]
        layer_config['batch_input_shape'][0] = op.dims.inp[0]
        layer_config['batch_input_shape'][1] = op.dims.inp[2]
        layer_config['batch_input_shape'][2] = op.dims.inp[3]
        layer_config['batch_input_shape'][3] = op.dims.inp[1]
        layer_config['name'] = layer_name
        layer_config['padding'] = 'same'  # TODO, op.tvm_node.attrs.padding
        layer_config['strides'] = convert_IntImm_array(op.tvm_node.attrs.strides)
        layer_config['dilation_rate'] = convert_IntImm_array(op.tvm_node.attrs.dilation)
        layer_config['pool_size'] = convert_IntImm_array(op.tvm_node.attrs.pool_size)
        ret['config'] = layer_config
        return ret, None, 0

    def _generate_hls_pool2d(self, op, layer_name, next_op=None, next_next_op=None):
        ret = {'class_name': 'MaxPooling2D', 'config': {}}
        layer_config = {'data_format': 'channels_last', 'dtype': op.used_dtype.name, 'trainable': False}
        layer_config['batch_input_shape'] = [0, 0, 0, 0]
        layer_config['batch_input_shape'][0] = op.dims.inp[0]
        layer_config['batch_input_shape'][1] = op.dims.inp[2]
        layer_config['batch_input_shape'][2] = op.dims.inp[3]
        layer_config['batch_input_shape'][3] = op.dims.inp[1]
        layer_config['name'] = layer_name
        layer_config['padding'] = 'same'  # TODO, op.tvm_node.attrs.padding
        layer_config['strides'] = convert_IntImm_array(op.tvm_node.attrs.strides)
        layer_config['dilation_rate'] = convert_IntImm_array(op.tvm_node.attrs.dilation)
        layer_config['pool_size'] = convert_IntImm_array(op.tvm_node.attrs.pool_size)
        ret['config'] = layer_config
        return ret, None, 0

    def _generate_hls_globalPool1d(self, op, layer_name, next_op=None, next_next_op=None):
        return

    def _generate_hls_globalPool2d(self, op, layer_name, next_op=None, next_next_op=None):
        return

    def _generatae_hls_prelu(self, op, layer_name, next_op=None, next_next_op=None):
        return

    def _generate_hls_parAct(self, op, layer_name, next_op=None, next_next_op=None):
        return

    def _generate_hls_softmax(self, op, layer_name, next_op=None, next_next_op=None):
        return

    def _generate_hls_dense(self, op, layer_name, next_op=None, next_next_op=None):
        ret1 = {'class_name': 'Dense',
                'config': {'activation': 'relu', 'activity_regularizer': None, 'bias_constraint': None,
                           'bias_initializer': {'class_name': 'VarianceScaling',
                                                'config': {'distribution': 'uniform', 'mode': 'fan_in', 'scale': 1.0,
                                                           'seed': None}},
                           'bias_regularizer': None, 'kernel_constraint': None,
                           'kernel_initializer': {'class_name': 'VarianceScaling',
                                                  'config': {'distribution': 'uniform', 'mode': 'fan_in', 'scale': 1.0,
                                                             'seed': None}},
                           'kernel_regularizer': {'class_name': 'L1L2', 'config': {'l1': 0.0, 'l2': 0.0}},
                           'name': 'fc1_relu',
                           'trainable': True,
                           'units': 64, 'use_bias': True}, 'inbound_nodes': [[['input_1', 0, 0, {}]]],
                'name': 'fc1_relu'}
        ret = {'class_name': 'Dense', 'config': {}}
        # 'config' must contain 'data_format' 'channels_first'
        layer_config = {'data_format': 'channels_last', 'dtype': op.used_dtype.name, 'trainable': False}
        # assert op.tvm_node.attrs.data_layout == 'NCHW'
        layer_config['name'] = layer_name
        layer_config['kernel_constraint'] = None
        layer_config['kernel_initializer'] = None
        layer_config['kernel_regularizer'] = None
        if len(op.dims.inp) == 4:
            layer_config['batch_input_shape'] = [0, 0, 0, 0]
            layer_config['batch_input_shape'][0] = op.dims.inp[0]
            layer_config['batch_input_shape'][1] = op.dims.inp[2]
            layer_config['batch_input_shape'][2] = op.dims.inp[3]
            layer_config['batch_input_shape'][3] = op.dims.inp[1]
        else:
            layer_config['batch_input_shape'] = op.dims.inp
        if op.tvm_node.attrs.units is not None:
            layer_config['units'] = op.tvm_node.attrs.units.value
        # else: not necessary?
        #     layer_config['units'] = 0
        layer_config['use_bias'] = False
        # further: name, class_name, activation, use_bias and 'epsilon'?

        # data must have var names 'kernel' and 'bias' (also 'depthwise_kernel?')
        data = {'layer_name': layer_name, 'variables': {}, 'data_format': {}}
        data['variables']['kernel'] = op.tvm_args['by_position'][1]['ref'].data.numpy()
        data['data_format']['kernel'] = 'channels_first'

        consumed_opt_ops = 0
        if next_op is not None and (next_op.op_call == 'add' or next_op.op_call == 'nn.bias_add'):
            layer_config['use_bias'] = True
            layer_config['bias_constraint'] = None
            layer_config['bias_initializer'] = None
            layer_config['bias_regularizer'] = None
            consumed_opt_ops += 1
            if next_op.op_call == 'add':
                # it is still called bias
                data['variables']['bias'] = next_op.tvm_args['by_position'][1]['ref'].data.numpy()
            else:
                data['variables']['bias'] = next_op.tvm_args['by_position'][1]['ref'].data.numpy()
            data['data_format']['bias'] = 'channels_first'
        else:
            layer_config['use_bias'] = False
            layer_config['bias_constraint'] = None
            layer_config['bias_initializer'] = None
            layer_config['bias_regularizer'] = None

        if next_next_op is not None and next_next_op.op_call == 'nn.relu':
            layer_config['activity_regularizer'] = None
            layer_config['activation'] = 'relu'
            consumed_opt_ops += 1
        elif next_next_op is not None and next_next_op.op_call == 'nn.softmax':
            layer_config['activity_regularizer'] = None
            layer_config['activation'] = 'softmax'
            consumed_opt_ops += 1
        elif next_next_op is not None and (next_next_op.op_call == 'nn.tanh' or next_next_op.op_call == 'tanh'):
            layer_config['activity_regularizer'] = None
            layer_config['activation'] = 'tanh'
            consumed_opt_ops += 1
        else:
            layer_config['activity_regularizer'] = None
            # conv_config['activation'] = None  # don't put the key in

        # assemble dict
        ret['config'] = layer_config
        return ret, data, consumed_opt_ops

    def _generate_hls_batchNorm(self, op, layer_name, next_op=None, next_next_op=None):
        return

    def _generate_hls_padding(self, op, layer_name, next_op=None, next_next_op=None):
        # including ZeroPadding
        return

    def _generate_hls_biasAdd(self, op, layer_name, next_op=None, next_next_op=None):
        return

    def _generate_hls_act(self, op, layer_name, next_op=None, next_next_op=None):
        return

    def _generate_hls_merge(self, op, layer_name, next_op=None, next_next_op=None):
        return

    def _generate_hls_transpose(self, op, layer_name, next_op=None, next_next_op=None):
        return

    def _generate_hls_reshape(self, op, layer_name, next_op=None, next_next_op=None):
        return

    def _generate_hls_flatten(self, op, layer_name, next_op=None, next_next_op=None):
        ret1 = {'class_name': 'Flatten', 'config': {'name': 'flatten_1', 'trainable': True}}
        ret = {'class_name': 'Flatten',
               'config': {'data_format': 'channels_first', 'dtype': op.used_dtype.name, 'trainable': False,
                          'batch_input_shape': op.dims.inp, 'name': layer_name}}
        # to much parameter don't harm?
        return ret, None, 0

    def _generate_hls_dropout(self, op, layer_name, next_op=None, next_next_op=None):
        # see e.g. https://github.com/fastmachinelearning/hls4ml/blob/e804cfc6bbadd9b64857e2dbd2459a5b7200ffb7/hls4ml/converters/onnx_to_hls.py#L286
        ret = {'class_name': 'Dropout', 'config': {'name': layer_name, 'rate': 0.25, 'trainable': False}}
        # TODO: smth like op.tvm_node.attrs.rate
        #  but will anyhow be ignored
        return ret, None, 0
