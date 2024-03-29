#  /*******************************************************************************
#   * Copyright 2019 -- 2024 IBM Corporation
#   *
#   * Licensed under the Apache License, Version 2.0 (the "License");
#   * you may not use this file except in compliance with the License.
#   * You may obtain a copy of the License at
#   *
#   *     http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS,
#   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   * See the License for the specific language governing permissions and
#   * limitations under the License.
#  *******************************************************************************/
#

#  *
#  *                       cloudFPGA
#  *    =============================================
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        DOSA OSG to implement vhdl4cnn on FPGAs
#  *
#  *        This OSG is based on https://github.com/DreamIP/haddoc2,
#  *        licensed under Apache 2.0.
#  *        Also code snippets were copied for the build methods.
#  *        The original repository, including all license information,
#  *        can be also found under `backend/3rd_party_libs/`.
#  *
#  *
import os
import numpy as np
import tvm
import tvm.relay as relay
import math
import json


import gradatim.lib.singleton as dosa_singleton
from gradatim.backend.buildTools.BaseBuild import HwBuildTopVhdl
from gradatim.backend.buildTools.cFBuild1 import cFBuild1
from gradatim.backend.codeGen.Haddoc2Wrapper import Haddoc2Wrapper
from gradatim.backend.operatorSets.BaseOSG import BaseOSG
from gradatim.backend.devices.dosa_device import DosaHwClasses
from gradatim.lib.dosa_dtype import get_bitwidth_of_DosaDtype, DosaDtype, DosaDtype_is_signed, \
    data_array_convert_to_DosaDtype
from gradatim.lib.util import BrickImplTypes
from gradatim.middleend.archGen.ArchBrick import ArchBrick
from gradatim.backend.operatorSets.relay_ops import op as relay_op_list
import gradatim.backend.operatorSets.lib.vhdl4cnn.paramsParsing as paramParsing
import gradatim.backend.operatorSets.lib.vhdl4cnn.topologyParsing as topologyParsing
from gradatim.backend.codeGen.WrapperInterfaces import InterfaceAxisFifo, wrapper_default_interface_bitwidth
from gradatim.middleend.archGen.OperationContract import OperationContract
from gradatim.backend.operatorSets.lib.util import get_avg_util_dict_bytes_based, get_share_of_FPGA_resources
import gradatim.lib.units as units


__filedir__ = os.path.dirname(os.path.abspath(__file__))
__db_path__ = __filedir__ + '/osg_impl_db.json'
_part_conv_ = 0.8
_part_bias_ = 0.15
_part_act_ = 0.05


class vhdl4cnnOSG(BaseOSG):

    def __init__(self):
        super().__init__('VHDL4CNN OSG', [DosaHwClasses.FPGA_generic, DosaHwClasses.FPGA_xilinx],
                         [DosaDtype.int2, DosaDtype.int3, DosaDtype.int4, DosaDtype.int8,
                          DosaDtype.int16, DosaDtype.int32],
                         # not uint8!
                         [BrickImplTypes.STREAM])
        self.priority = 99
        me_abs_dir = os.path.dirname(os.path.realpath(__file__))
        self.my_hdl_lib_folder = os.path.abspath(me_abs_dir + '/../third_party_libs/vhdl4cnn/lib/hdl/')
        self.my_hdl_template_folder = os.path.abspath(me_abs_dir + '/../third_party_libs/vhdl4cnn/lib/templates/')
        self.existing_layer_names = []
        self.util_db = {}
        self.avg_util_dict = {}
        self.pipeline_tensor_store = 2
        self._block_multi_threshold_used_id = 0

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
        if consider_paramB:
            bytes_total = paramB
        else:
            bytes_total = inpB
        bytes_total *= custom_byte_factor
        ret_dict['LUTLOG'] = res_dict['LUTLOG'] * bytes_total
        ret_dict['LUTMEM'] = res_dict['LUTMEM'] * bytes_total
        ret_dict['Registers'] = res_dict['Registers'] * bytes_total
        ret_dict['BRAM'] = res_dict['BRAM'] * bytes_total
        ret_dict['DSPs'] = res_dict['DSPs'] * bytes_total
        ret_dict['latency_lim_per_tensor_cycl'] = res_dict['latency_lim_per_tensor_cycl'] * inpB
        wrapper_dict = {}
        wrapper_dict['LUTLOG'] = res_dict['wrapper']['LUTLOG'] * bytes_total
        wrapper_dict['LUTMEM'] = res_dict['wrapper']['LUTMEM'] * bytes_total
        wrapper_dict['Registers'] = res_dict['wrapper']['Registers'] * bytes_total
        wrapper_dict['BRAM'] = res_dict['wrapper']['BRAM'] * bytes_total
        wrapper_dict['DSPs'] = res_dict['wrapper']['DSPs'] * bytes_total
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
            # in DOSA extension of TVM
            elif 'multi_threshold' in e:
                self.relay2osg['nn'][e] = self._generate_multithreshold, self._predict_multithreshold
        for e in self.relay2osg:
            if type(self.relay2osg[e]) == dict:
                continue
            elif 'tanh' in e:
                self.relay2osg[e] = self._generate_hdl_tanh_instance, self._predict_tanh

    def _copy_hdl_lib(self, target_hdl_dir, block_id):
        os.system("cp -f {}/* {}/".format(self.my_hdl_lib_folder, target_hdl_dir))
        # overwrite cnn_types, due to other lib name
        with open(os.path.join(self.my_hdl_lib_folder, 'cnn_types.vhd'), 'r') as in_file, \
                open(os.path.join(target_hdl_dir, 'cnn_types.vhd'), 'w') as out_file:
            line_num = 1
            for line in in_file.readlines():
                if line_num == 54:
                    outline = 'use work.bitwidths_b{}.all;  -- automatically adjusted by DOSA\n'.format(block_id)
                else:
                    outline = line
                out_file.write(outline)
                line_num += 1
        return 0

    def _get_and_increment_multi_threshold_used_id(self):
        ret = self._block_multi_threshold_used_id
        self._block_multi_threshold_used_id += 1
        return ret

    def build_block(self, arch_block, build_tool, selected_contracts):
        self._block_multi_threshold_used_id = 0
        assert isinstance(build_tool, HwBuildTopVhdl)
        if isinstance(build_tool, cFBuild1):
            # for cF, everything under hdl/ will be included
            hybrid_dir = build_tool.add_ip_dir(arch_block.block_uuid, hybrid=True)
            used_vhdl_dir_path = hybrid_dir[0]
            used_hls_dir_path = hybrid_dir[1]
        else:
            used_vhdl_dir_path = build_tool.add_ip_dir(arch_block.block_uuid)
            used_hls_dir_path = used_vhdl_dir_path
        # TODO: add multiple variants
        # assert selected_contracts.osg_intern_id == 'basis'
        # first, copy all lib files
        # use global_vhdl_dir to have it only once per node --> no, different cnn_types per block
        # self._copy_hdl_lib(build_tool.global_vhdl_dir)
        self._copy_hdl_lib(used_vhdl_dir_path, arch_block.block_uuid)
        # file names are in specific folder, don't need to match entity name
        paramFile = os.path.abspath(used_vhdl_dir_path + '/params.vhd')  # Configuration VHDL output
        topFile = os.path.abspath(used_vhdl_dir_path + '/cnn_process.vhd')  # Top level VHDL output
        bitwidthFile = os.path.abspath(used_vhdl_dir_path + '/bitwidths.vhd')  # Bitwidth  VHDL output
        multiThresholdContainer_file = os.path.abspath(used_vhdl_dir_path + '/MultiThresholdOperation.vhd')  # Bitwidth  VHDL output

        used_dtype = DosaDtype.UNKNOWN
        layer_names_by_op_id = {}
        layer_names_ordered = []
        input_dims_by_op_id = {}
        input_dims_ordered = []
        ops_implemented_ordered = []
        wrapper_flatten_op = None
        wrapper_first_brick = None
        wrapper_last_brick = None
        total_delay = 1  # first internal register after DynInput
        multi_threshold_op_list = []
        # as haddoc, we first take care of the params
        with open(paramFile, 'w') as vhdlf:
            paramParsing.write_fileHead(vhdlf, arch_block.block_uuid)
            # now, iterate over bricks
            for bb in arch_block.brick_list:
                if wrapper_first_brick is None:
                    wrapper_first_brick = bb
                wrapper_last_brick = bb
                skip_i = []
                # for op in bb.local_op_iter_gen():
                for op_i in bb.ops.keys():
                    op = bb.ops[op_i]
                    layer_name = self._create_unique_layer_name(op.name)
                    if 'multi_threshold' in op.op_call:
                        multi_threshold_op_list.append(op)
                        continue
                    if op_i in skip_i:
                        continue

                    next_i = op_i + 1
                    next_op = None
                    if next_i in bb.ops.keys():
                        next_op = bb.ops[next_i]
                    next_next_i = op_i + 2
                    next_next_op = None
                    if next_next_i in bb.ops.keys():
                        next_next_op = bb.ops[next_next_i]

                    if used_dtype == DosaDtype.UNKNOWN:
                        used_dtype = op.used_dtype
                    elif used_dtype != op.used_dtype:
                        print("[DOSA:OSG:ERROR] Haddoc supports only one bit width per block. Trying to ignore...")

                    if 'flatten' in op.op_call:
                        # TODO: if flatten is in the middle of a longer block?
                        # layer name will be ignored
                        wrapper_flatten_op = op
                        if op_i + 1 != len(bb.ops):
                            print("[DOSA:vhdl4cnnOSG:ERROR] flatten is only supported as last layer!. STOP.")
                            exit(1)
                    else:
                        osg_func = self._get_osg_func(op.op_call)
                        mod_op, consumed_opt_ops, add_delay = osg_func(op, vhdlf, layer_name, next_op, next_next_op)

                        total_delay += add_delay + 1
                        layer_names_by_op_id[op.global_op_id] = layer_name
                        layer_names_ordered.append(layer_name)
                        input_dims_by_op_id[op.global_op_id] = op.dims.inp
                        input_dims_ordered.append(op.dims.inp)
                        if mod_op is None:
                            ops_implemented_ordered.append(op)
                        else:
                            ops_implemented_ordered.append(mod_op)
                        if consumed_opt_ops >= 1:
                            skip_i.append(next_i)
                        if consumed_opt_ops >= 2:
                            skip_i.append(next_next_i)

                    # if 'pool2d' in op.op_call:
                    #     layer_names_by_op_id[op.global_op_id] = layer_name
                    #     layer_names_ordered.append(layer_name)
                    #     input_dims_by_op_id[op.global_op_id] = op.dims.inp
                    #     input_dims_ordered.append(op.dims.inp)
                    #     ops_implemented_ordered.append(op)
                    #     self._param_parse_pool(op, vhdlf, layer_name)
                    # elif 'conv2d' in op.op_call:
                    #     # TODO map 1d correctly
                    #     layer_names_by_op_id[op.global_op_id] = layer_name
                    #     layer_names_ordered.append(layer_name)
                    #     input_dims_by_op_id[op.global_op_id] = op.dims.inp
                    #     input_dims_ordered.append(op.dims.inp)
                    #     ops_implemented_ordered.append(op)
                    #     if next_op is not None and 'bias' not in next_op.op_call:
                    #         # useless -> None again
                    #         next_op = None
                    #     self._param_parse_conv(op, vhdlf, layer_name, next_op)
                    # elif 'flatten' in op.op_call:
                    #     # TODO: if flatten is in the middle of a longer block?
                    #     # layer name will be ignored
                    #     wrapper_flatten_op = op
                    #     if op_i + 1 != len(bb.ops):
                    #         print("[DOSA:vhdl4cnnOSG:ERROR] flatten is only supported as last layer!. STOP.")
                    #         exit(1)
                    # elif 'dropout' in op.op_call:
                    #     print("[DOSA:OSG:ERROR] Not yet implemented!. STOP.")
                    #     exit(1)
                    # else:
                    #     print("[DOSA:OSG:ERROR] Not yet implemented!. STOP.")
                    #     exit(1)

            paramParsing.write_fileEnd(vhdlf)
        # haddoc only supports signed datatypes
        assert DosaDtype_is_signed(used_dtype)

        # then, we do the bitwidth
        used_bit_width = get_bitwidth_of_DosaDtype(used_dtype)
        input_dim = input_dims_ordered[0]
        input_bit_width = used_bit_width
        # TODO: make dynamic
        input_bit_width *= input_dim[1]  # num channels
        if input_dim[0] != 1:
            print("[DOSA:OSG:ERROR] Haddoc2 only supports models with batch_size 1 (currently).")
            exit(-1)
        output_dim = ops_implemented_ordered[-1].dims.out
        output_bit_width = used_bit_width
        # TODO: make dynamic
        output_bit_width *= output_dim[1]
        # default prod_width
        prod_width = used_bit_width * 2 + 1  # sum of mult (adder tree...)
        # determine prod_width based on brevitas results
        for op in multi_threshold_op_list:
            layer_data = op.tvm_args['by_position'][1]['ref'].data.numpy()
            max_number = max(np.max(layer_data), abs(np.min(layer_data)))
            max_threshold_bitwidth = int(np.ceil(np.log2(max_number))) + 1  # +1 for always signed!
            if max_threshold_bitwidth > prod_width:
                prod_width = max_threshold_bitwidth
        print(f"[DOSA:vhdl4cnnOSG:DEBUG] determined prod_with: {prod_width}")
        self._generate_bitwidth(bitwidthFile, arch_block.block_uuid, used_bit_width, input_bit_width, output_bit_width,
                                prod_width)
        wrapper_first_op = ops_implemented_ordered[0]
        wrapper_last_op = None
        # next, create global multi_threshold
        if self._block_multi_threshold_used_id > 0:
            self._generate_multi_threshold_container(multi_threshold_op_list, multiThresholdContainer_file, prod_width)
        # finally, create the topology
        with open(topFile, 'w') as topf:
            topologyParsing.WriteLibs(topf, arch_block.block_uuid)
            topologyParsing.WriteEntity(topf, arch_block.block_uuid,
                                        layer_names_by_op_id[wrapper_first_op.global_op_id])
            topologyParsing.WriteArchitecutreHead(topf, arch_block.block_uuid)
            # we always need an input
            input_layer_name = "haddoc2_osg_input"
            topf.write(" -- Signals\n")
            topologyParsing.WriteInputSignal(topf, input_layer_name, layer_names_ordered[0])
            # for all other layers
            # for op in bb.local_op_iter_gen():
            for op in ops_implemented_ordered:
                layer_name = layer_names_by_op_id[op.global_op_id]
                topologyParsing.WriteLayerSignal(topf, layer_name)
            topf.write("\n")
            # then, components
            topologyParsing.WriteComponents(topf)
            topf.write(" -- Instances\n")
            topf.write("begin\n")
            # again, input first
            topologyParsing.InstanceDynInputLayer(topf, input_layer_name, layer_names_ordered[0], input_bit_width)
            # for all other layers
            previous_layer_name = input_layer_name
            # for op in bb.local_op_iter_gen():
            for op in ops_implemented_ordered:
                wrapper_last_op = op
                layer_name = layer_names_by_op_id[op.global_op_id]
                if 'conv2d' in op.op_call:
                    use_relu_activation = False
                    use_tanh_activation = False
                    use_multi_threshold_activation = False
                    if op.haddoc_activation == 'tanh':
                        use_tanh_activation = True
                    elif op.haddoc_activation == 'relu':
                        use_relu_activation = True
                    elif op.haddoc_activation == 'multi_threshold':
                        use_multi_threshold_activation = True
                    # else: default...
                    topologyParsing.InstanceConvLayer(topf, layer_name, previous_layer_name,
                                                      use_relu_activation=use_relu_activation,
                                                      use_tanh_activation=use_tanh_activation,
                                                      use_multithreshold_activation=use_multi_threshold_activation,
                                                      multi_threshold_id=op.multi_threshold_used_id)
                elif 'pool2d' in op.op_call:
                    topologyParsing.InstancePoolLayer(topf, layer_name, previous_layer_name)
                else:
                    continue
                previous_layer_name = layer_name
            topologyParsing.InstanceDynOutputLayer(topf, previous_layer_name)
            topologyParsing.WriteArchitectureEnd(topf)

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

        block_wrapper = Haddoc2Wrapper(arch_block.block_uuid, wrapper_first_op.dims.inp, wrapper_last_op.dims.out,
                                       used_bit_width, if_in_bitw, if_out_bitw, used_hls_dir_path, wrapper_flatten_op,
                                       len(ops_implemented_ordered),
                                       layer_names_by_op_id[wrapper_first_op.global_op_id], total_delay)
        block_wrapper.generate_haddoc2_wrapper()
        build_tool.add_makefile_entry(used_hls_dir_path, 'all')
        wrapper_inst_tcl = block_wrapper.get_tcl_lines_wrapper_inst('IP Core to connect DOSA infrastructure with '
                                                                    'Haddoc2 Layers')
        build_tool.add_tcl_entry(wrapper_inst_tcl)
        wrapper_decl = block_wrapper.get_wrapper_vhdl_decl_lines()
        wrapper_inst_tmpl = block_wrapper.get_vhdl_inst_tmpl()

        lib_lines = ['bitwidths_b{}.all'.format(arch_block.block_uuid), 'cnn_types.all',
                     'params_b{}.all'.format(arch_block.block_uuid)]
        build_tool.topVhdl.add_lib_include('work', lib_lines)
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
        print("[DOSA:Build:ERROR] Haddoc2 OSG was asked to build an engine container, but it can't. STOP.")
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

    # def _generate_hdl_conv_instance(self, todo):
    #     print("[DOSA:Build:ERROR] Currently, Haddoc2 operators can only be implemented block-wise " +
    #           "(i.e. use build_block). IGNORING.")
    #     return

    # def _generate_hdl_pool_instance(self, todo):
    #     print("[DOSA:Build:ERROR] Currently, Haddoc2 operators can only be implemented block-wise " +
    #           "(i.e. use build_block). IGNORING.")
    #     return

    def _predict_tanh(self, op, target_hw, impl_type):
        if impl_type != BrickImplTypes.STREAM or \
                (target_hw.hw_class != DosaHwClasses.FPGA_xilinx and target_hw.hw_class != DosaHwClasses.FPGA_generic):
            return None
        util_dict, wrapper_dict, used_fallback = self._get_impl_prediction('tanh', op.input_bytes, 0,
                                                                           target_hw, consider_paramB=False)
        # using relu as fallback
        if used_fallback:
            util_dict, wrapper_dict, _ = self._get_impl_prediction('relu', op.input_bytes, 0,
                                                                   target_hw, consider_paramB=False)
        util_dict['LUTLOG'] *= _part_act_
        util_dict['LUTMEM'] *= _part_act_
        util_dict['Registers'] *= _part_act_
        util_dict['BRAM'] *= _part_act_
        util_dict['DSPs'] *= _part_act_
        wrapper_dict['LUTLOG'] *= _part_act_
        wrapper_dict['LUTMEM'] *= _part_act_
        wrapper_dict['Registers'] *= _part_act_
        wrapper_dict['BRAM'] *= _part_act_
        wrapper_dict['DSPs'] *= _part_act_
        proc_share = get_share_of_FPGA_resources(target_hw.get_resource_dict()['FPGA_utility'], util_dict)
        wrapper_share = get_share_of_FPGA_resources(target_hw.get_resource_dict()['FPGA_utility'], wrapper_dict)
        # proc_comp_share = (proc_share['LUTLOG'] + proc_share['DSPs']) / 2
        # TODO: equal weights (i.e. just /2), makes sense?
        proc_comp_share = proc_share['LUTLOG']  # we know we hardly use DSPs..
        proc_mem_share = (proc_share['LUTMEM'] + proc_share['Registers'] + proc_share['BRAM']) / 3
        # wrapper_comp_share = (wrapper_share['LUTLOG'] + wrapper_share['DSPs']) / 2
        wrapper_comp_share = wrapper_share['LUTLOG']
        wrapper_mem_share = (wrapper_share['LUTMEM'] + wrapper_share['Registers'] + wrapper_share['BRAM']) / 3
        # latency_ns = math.ceil(util_dict['latency_lim_per_tensor_cycl'] * _part_act_ *
        #                        target_hw.get_performance_dict()['fpga_clk_ns'])
        latency_ns = 1 * target_hw.get_performance_dict()['fpga_clk_ns']  # more or less for free
        iter_hz = 1 / (latency_ns * units.nanoU)
        offer = OperationContract(op, target_hw, self, BrickImplTypes.STREAM, iter_hz, proc_comp_share, proc_mem_share,
                                  'basis', wrapper_comp_share, wrapper_mem_share, proc_share, wrapper_share)
        return offer

    def _generate_hdl_tanh_instance(self, todo):
        # is merged with ConvLayer.vhd?
        # but separate TanH exist, so could be added
        print("[DOSA:Build:ERROR] Currently, Haddoc2 operators can only be implemented block-wise " +
              "(i.e. use build_block). IGNORING.")
        return

    def _predict_flatten_drop(self, op, target_hw, impl_type):
        if impl_type != BrickImplTypes.STREAM or \
                (target_hw.hw_class != DosaHwClasses.FPGA_xilinx and target_hw.hw_class != DosaHwClasses.FPGA_generic):
            return None
        offer = OperationContract(op, target_hw, self, BrickImplTypes.STREAM, BaseOSG._pseudo_infinity_, 0.0, 0.0,
                                  'basis', 0.0, 0.0)
        return offer

    def _generate_hdl_flatten_instance(self, todo):
        print("[DOSA:Build:ERROR] Currently, Haddoc2 operators can only be implemented block-wise " +
              "(i.e. use build_block). IGNORING.")
        return

    def _generate_hdl_dropout_instance(self, todo):
        print("[DOSA:Build:ERROR] Currently, Haddoc2 operators can only be implemented block-wise " +
              "(i.e. use build_block). IGNORING.")
        return

    def _predict_bias(self, op, target_hw, impl_type):
        if impl_type != BrickImplTypes.STREAM or \
                (target_hw.hw_class != DosaHwClasses.FPGA_xilinx and target_hw.hw_class != DosaHwClasses.FPGA_generic):
            return None
        util_dict, wrapper_dict, used_fallback = self._get_impl_prediction('bias_add', op.input_bytes,
                                                                           op.parameter_bytes,
                                                                           target_hw, consider_paramB=True)
        util_dict['LUTLOG'] *= _part_bias_
        util_dict['LUTMEM'] *= _part_bias_
        util_dict['Registers'] *= _part_bias_
        util_dict['BRAM'] *= _part_bias_
        util_dict['DSPs'] *= _part_bias_
        wrapper_dict['LUTLOG'] *= _part_bias_
        wrapper_dict['LUTMEM'] *= _part_bias_
        wrapper_dict['Registers'] *= _part_bias_
        wrapper_dict['BRAM'] *= _part_bias_
        wrapper_dict['DSPs'] *= _part_bias_
        proc_share = get_share_of_FPGA_resources(target_hw.get_resource_dict()['FPGA_utility'], util_dict)
        wrapper_share = get_share_of_FPGA_resources(target_hw.get_resource_dict()['FPGA_utility'], wrapper_dict)
        # proc_comp_share = (proc_share['LUTLOG'] + proc_share['DSPs']) / 2
        proc_comp_share = proc_share['LUTLOG']  # we know we hardly use DSPs..
        proc_mem_share = (proc_share['LUTMEM'] + proc_share['Registers'] + proc_share['BRAM']) / 3
        # wrapper_comp_share = (wrapper_share['LUTLOG'] + wrapper_share['DSPs']) / 2
        wrapper_comp_share = wrapper_share['LUTLOG']
        wrapper_mem_share = (wrapper_share['LUTMEM'] + wrapper_share['Registers'] + wrapper_share['BRAM']) / 3
        # latency_ns = util_dict['latency_lim_per_tensor_cycl'] * _part_bias_ * \
        #              target_hw.get_performance_dict()['fpga_clk_ns']
        latency_ns = 1 * target_hw.get_performance_dict()['fpga_clk_ns']  # more or less for free
        iter_hz = 1 / (latency_ns * units.nanoU)
        offer = OperationContract(op, target_hw, self, BrickImplTypes.STREAM, iter_hz, proc_comp_share, proc_mem_share,
                                  'basis', wrapper_comp_share, wrapper_mem_share, proc_share, wrapper_share)
        return offer

    def _generate_hdl_biasAdd(self, todo):
        print("[DOSA:Build:ERROR] Currently, Haddoc2 operators can only be implemented block-wise " +
              "(i.e. use build_block). IGNORING.")
        return

    def _predict_relu(self, op, target_hw, impl_type):
        if impl_type != BrickImplTypes.STREAM or \
                (target_hw.hw_class != DosaHwClasses.FPGA_xilinx and target_hw.hw_class != DosaHwClasses.FPGA_generic):
            return None
        util_dict, wrapper_dict, used_fallback = self._get_impl_prediction('relu', op.input_bytes, 0,
                                                                           target_hw, consider_paramB=False)
        if used_fallback:
            util_dict, wrapper_dict, _ = self._get_impl_prediction('tanh', op.input_bytes, 0,
                                                                   target_hw, consider_paramB=False)
        util_dict['LUTLOG'] *= _part_act_
        util_dict['LUTMEM'] *= _part_act_
        util_dict['Registers'] *= _part_act_
        util_dict['BRAM'] *= _part_act_
        util_dict['DSPs'] *= _part_act_
        wrapper_dict['LUTLOG'] *= _part_act_
        wrapper_dict['LUTMEM'] *= _part_act_
        wrapper_dict['Registers'] *= _part_act_
        wrapper_dict['BRAM'] *= _part_act_
        wrapper_dict['DSPs'] *= _part_act_
        proc_share = get_share_of_FPGA_resources(target_hw.get_resource_dict()['FPGA_utility'], util_dict)
        wrapper_share = get_share_of_FPGA_resources(target_hw.get_resource_dict()['FPGA_utility'], wrapper_dict)
        # proc_comp_share = (proc_share['LUTLOG'] + proc_share['DSPs']) / 2
        proc_comp_share = proc_share['LUTLOG']  # we know we hardly use DSPs..
        proc_mem_share = (proc_share['LUTMEM'] + proc_share['Registers'] + proc_share['BRAM']) / 3
        # wrapper_comp_share = (wrapper_share['LUTLOG'] + wrapper_share['DSPs']) / 2
        wrapper_comp_share = wrapper_share['LUTLOG']  # we know we hardly use DSPs...
        wrapper_mem_share = (wrapper_share['LUTMEM'] + wrapper_share['Registers'] + wrapper_share['BRAM']) / 3
        # latency_ns = util_dict['latency_lim_per_tensor_cycl'] * _part_act_ \
        #              * target_hw.get_performance_dict()['fpga_clk_ns']
        latency_ns = 1 * target_hw.get_performance_dict()['fpga_clk_ns']  # more or less for free
        iter_hz = 1 / (latency_ns * units.nanoU)
        offer = OperationContract(op, target_hw, self, BrickImplTypes.STREAM, iter_hz, proc_comp_share, proc_mem_share,
                                  'basis', wrapper_comp_share, wrapper_mem_share, proc_share, wrapper_share)
        return offer

    def _generate_hdl_relu(self, todo):
        print("[DOSA:Build:ERROR] Currently, Haddoc2 operators can only be implemented block-wise " +
              "(i.e. use build_block). IGNORING.")
        return

    def _create_unique_layer_name(self, op_name):
        base_str = op_name.replace('.', '_').replace('/', 'of').replace(' ', '') \
            .replace(',', '_').replace('-', '__')
        name_cnt = 1
        while base_str in self.existing_layer_names:
            base_str += "_{}".format(name_cnt)
            name_cnt += 1
        self.existing_layer_names.append(base_str)
        return base_str

    def _predict_conv(self, op, target_hw, impl_type):
        if impl_type != BrickImplTypes.STREAM or \
                (target_hw.hw_class != DosaHwClasses.FPGA_xilinx and target_hw.hw_class != DosaHwClasses.FPGA_generic):
            return None
        util_dict, wrapper_dict, used_fallback = self._get_impl_prediction('conv2d', op.input_bytes, op.parameter_bytes,
                                                                           target_hw, consider_paramB=True,
                                                                           custom_byte_factor=1.6)
        util_dict['LUTLOG'] *= _part_conv_
        util_dict['LUTMEM'] *= _part_conv_
        util_dict['Registers'] *= _part_conv_
        util_dict['BRAM'] *= _part_conv_
        util_dict['DSPs'] *= _part_conv_
        wrapper_dict['LUTLOG'] *= _part_conv_
        wrapper_dict['LUTMEM'] *= _part_conv_
        wrapper_dict['Registers'] *= _part_conv_
        wrapper_dict['BRAM'] *= _part_conv_
        wrapper_dict['DSPs'] *= _part_conv_
        proc_share = get_share_of_FPGA_resources(target_hw.get_resource_dict()['FPGA_utility'], util_dict)
        wrapper_share = get_share_of_FPGA_resources(target_hw.get_resource_dict()['FPGA_utility'], wrapper_dict)
        # proc_comp_share = (proc_share['LUTLOG'] + proc_share['DSPs']) / 2
        proc_comp_share = proc_share['LUTLOG']  # we know we hardly use DSPs..
        proc_mem_share = (proc_share['LUTMEM'] + proc_share['Registers'] + proc_share['BRAM']) / 3
        # wrapper_comp_share = (wrapper_share['LUTLOG'] + wrapper_share['DSPs']) / 2
        wrapper_comp_share = wrapper_share['LUTLOG']  # we know we hardly use DSPs...
        wrapper_mem_share = (wrapper_share['LUTMEM'] + wrapper_share['Registers'] + wrapper_share['BRAM']) / 3
        # using complete latency, so that brick latency is correct
        input_data_width = op.dims.inp[2]  # image_width
        out_channel_num = op.dims.out[1]  # out_size
        in_channel_num = op.dims.inp[1]  # previous_layer_size
        kernel_size = op.dims.param[2]
        internal_delay = 2 + 1 + ((kernel_size - 1) * input_data_width) + kernel_size + 1 \
                         + 2 + 1
        used_cycles = internal_delay + np.prod(op.dims.inp)
        # latency_ns = util_dict['latency_lim_per_tensor_cycl'] * target_hw.get_performance_dict()['fpga_clk_ns']
        latency_ns = used_cycles * target_hw.get_performance_dict()['fpga_clk_ns']
        iter_hz = 1 / (latency_ns * units.nanoU)
        offer = OperationContract(op, target_hw, self, BrickImplTypes.STREAM, iter_hz, proc_comp_share, proc_mem_share,
                                  'basis', wrapper_comp_share, wrapper_mem_share, proc_share, wrapper_share)
        return offer

    def _param_parse_conv(self, op, target_fh, layer_name, next_op=None, next_next_op=None):
        consumed_opt_ops = 0
        assert op.tvm_node.attrs.data_layout == 'NCHW'
        out_channel_num = op.dims.out[1]  # out_size
        in_channel_num = op.dims.inp[1]  # previous_layer_size
        kernel_size = op.dims.param[2]
        assert kernel_size == op.dims.param[3]
        input_data_width = op.dims.inp[2]  # image_width
        assert input_data_width == op.dims.inp[3]
        assert isinstance(op.tvm_args['by_position'][1]['ref'], tvm.relay.expr.Constant)
        if op.need_to_cast_tvm_args:
            kernel_data = data_array_convert_to_DosaDtype(op.tvm_args['by_position'][1]['ref'].data.numpy(),
                                                          op.used_dtype,
                                                          data_already_scaled=dosa_singleton.config.quant.numbers_already_scaled,
                                                          numpy_array_type='int32')
        else:
            kernel_data = op.tvm_args['by_position'][1]['ref'].data.numpy()
        bias_data = np.zeros(out_channel_num, dtype=int)

        if next_op is not None and next_op.op_call == 'nn.bias_add':
            if isinstance(next_op.tvm_args['by_position'][1]['ref'], tvm.relay.expr.Constant):
                if op.need_to_cast_tvm_args:
                    bias_data = data_array_convert_to_DosaDtype(next_op.tvm_args['by_position'][1]['ref'].data.numpy(),
                                                                op.used_dtype,
                                                                data_already_scaled=dosa_singleton.config.quant.numbers_already_scaled,
                                                                numpy_array_type='int32')
                else:
                    bias_data = next_op.tvm_args['by_position'][1]['ref'].data.numpy()
            else:
                print("[DOSA:OSG:WARNING] Strange non-constant bias value for op {}".format(repr(next_op)))
            consumed_opt_ops += 1
        # in case there is no bias
        elif next_op is not None and next_next_op is None:
            next_next_op = next_op

        op.haddoc_activation = 'none'
        if next_next_op is not None and next_next_op.op_call == 'nn.relu':
            op.haddoc_activation = 'relu'
            consumed_opt_ops += 1
        elif next_next_op is not None and (next_next_op.op_call == 'nn.tanh' or next_next_op.op_call == 'tanh'):
            op.haddoc_activation = 'tanh'
            consumed_opt_ops += 1

        threshold_op = None
        op.multi_threshold_used_id = -1
        if next_op is not None and next_op.op_call == 'nn.multi_threshold':
            threshold_op = next_op
        elif next_next_op is not None and next_next_op.op_call == 'nn.multi_threshold':
            threshold_op = next_op
        if threshold_op is not None:
            op.haddoc_activation = 'multi_threshold'
            op.multi_threshold_used_id = self._get_and_increment_multi_threshold_used_id()
            consumed_opt_ops += 1

        nbits = get_bitwidth_of_DosaDtype(op.used_dtype)
        target_fh.write("--" + layer_name + "\n")
        paramParsing.write_image_width(layer_name, input_data_width, target_fh)
        paramParsing.write_in_size(layer_name, in_channel_num, target_fh)
        paramParsing.write_out_size(layer_name, out_channel_num, target_fh)
        paramParsing.write_kernel_size(layer_name, kernel_size, target_fh)
        paramParsing.write_bias_value(bias_data, layer_name, nbits, target_fh)
        paramParsing.write_kernel_value(kernel_data, layer_name, nbits, target_fh)
        target_fh.write("----------------------------------------------------------")
        target_fh.write("--------------------------------------------------------\n")
        internal_delay = 2 + 1 + ((kernel_size - 1) * input_data_width) + kernel_size + 1 \
                         + 2 + 1
        return op, consumed_opt_ops, internal_delay

    def _predict_pool(self, op, target_hw, impl_type):
        if impl_type != BrickImplTypes.STREAM or \
                (target_hw.hw_class != DosaHwClasses.FPGA_xilinx and target_hw.hw_class != DosaHwClasses.FPGA_generic):
            return None
        if 'global_avg_pool' in op.op_call:
            return None
        util_dict, wrapper_dict, used_fallback = self._get_impl_prediction('max_pool2d', op.input_bytes, 0,
                                                                           target_hw, consider_paramB=False)
        proc_share = get_share_of_FPGA_resources(target_hw.get_resource_dict()['FPGA_utility'], util_dict)
        wrapper_share = get_share_of_FPGA_resources(target_hw.get_resource_dict()['FPGA_utility'], wrapper_dict)
        # proc_comp_share = (proc_share['LUTLOG'] + proc_share['DSPs']) / 2
        proc_comp_share = proc_share['LUTLOG']  # we know we hardly use DSPs..
        proc_mem_share = (proc_share['LUTMEM'] + proc_share['Registers'] + proc_share['BRAM']) / 3
        # wrapper_comp_share = (wrapper_share['LUTLOG'] + wrapper_share['DSPs']) / 2
        wrapper_comp_share = wrapper_share['LUTLOG']  # we know we hardly use DSPs...
        wrapper_mem_share = (wrapper_share['LUTMEM'] + wrapper_share['Registers'] + wrapper_share['BRAM']) / 3
        input_data_width = op.dims.inp[2]  # image_width
        kernel_size = op.tvm_node.attrs.pool_size[0]
        internal_delay = 1 + input_data_width + kernel_size
        cycles_used = internal_delay + np.prod(op.dims.inp)
        # latency_ns = util_dict['latency_lim_per_tensor_cycl'] * target_hw.get_performance_dict()['fpga_clk_ns']
        latency_ns = cycles_used * target_hw.get_performance_dict()['fpga_clk_ns']
        iter_hz = 1 / (latency_ns * units.nanoU)
        offer = OperationContract(op, target_hw, self, BrickImplTypes.STREAM, iter_hz, proc_comp_share, proc_mem_share,
                                  'basis', wrapper_comp_share, wrapper_mem_share, proc_share, wrapper_share)
        return offer

    def _param_parse_pool(self, op, target_fh, layer_name, next_op=None, next_next_op=None):
        out_channel_num = op.dims.out[1]  # out_size
        assert out_channel_num == op.dims.inp[1]
        input_data_width = op.dims.inp[2]  # image_width
        assert input_data_width == op.dims.inp[3]
        kernel_size = op.tvm_node.attrs.pool_size[0]
        assert kernel_size == op.tvm_node.attrs.pool_size[1]  # only symmetric kernels are supported
        assert kernel_size == 2  # Haddoc2 supports only a subsampling factor of 4
        target_fh.write("--" + layer_name + "\n")
        paramParsing.write_image_width(layer_name, input_data_width, target_fh)
        paramParsing.write_in_size(layer_name, out_channel_num, target_fh)
        paramParsing.write_out_size(layer_name, out_channel_num, target_fh)
        paramParsing.write_kernel_size(layer_name, kernel_size, target_fh)
        target_fh.write("----------------------------------------------------------")
        target_fh.write("--------------------------------------------------------\n")
        internal_delay = 1 + input_data_width + kernel_size
        return None, 0, internal_delay

    def _generate_bitwidth(self, bitwidth_file, block_id, general_bitwidth, input_bitwidth, output_bitwidth, prod_width):
        with open(bitwidth_file, 'w') as f:
            f.write("-- this file is automatically generated by DOSA to instantiate the VHDL4CNN library --\n")
            f.write('library ieee;\n')
            f.write('  use ieee.std_logic_1164.all;\n')
            f.write('  use ieee.numeric_std.all;\n')
            f.write('  use ieee.math_real.all;\n')
            f.write(f'package bitwidths_b{block_id} is\n')
            f.write(f'  constant GENERAL_BITWIDTH : integer := {str(general_bitwidth)};\n')
            # f.write('  constant SUM_WIDTH        : integer := 3*GENERAL_BITWIDTH;\n')
            # f.write('  constant PROD_WIDTH        : integer := 2*GENERAL_BITWIDTH;\n')
            f.write(f'  constant PROD_WIDTH       : integer := {str(prod_width)};\n')
            f.write(f'  constant INPUT_BIT_WIDTH  : integer := {str(input_bitwidth)};\n')
            f.write(f'  constant OUTPUT_BITWIDTH  : integer := {str(output_bitwidth)};\n')
            f.write(f'end bitwidths_b{block_id};\n')
        return

    def _predict_multithreshold(self, op, target_hw, impl_type):
        if impl_type != BrickImplTypes.STREAM or \
                (target_hw.hw_class != DosaHwClasses.FPGA_xilinx and target_hw.hw_class != DosaHwClasses.FPGA_generic):
            return None
        # TODO: update db?
        util_dict, wrapper_dict, used_fallback = self._get_impl_prediction('max_pool2d', op.input_bytes, 0,
                                                                           target_hw, consider_paramB=False)
        proc_share = get_share_of_FPGA_resources(target_hw.get_resource_dict()['FPGA_utility'], util_dict)
        wrapper_share = get_share_of_FPGA_resources(target_hw.get_resource_dict()['FPGA_utility'], wrapper_dict)
        # proc_comp_share = (proc_share['LUTLOG'] + proc_share['DSPs']) / 2
        proc_comp_share = proc_share['LUTLOG']  # we know we hardly use DSPs..
        proc_mem_share = (proc_share['LUTMEM'] + proc_share['Registers'] + proc_share['BRAM']) / 3
        # wrapper_comp_share = (wrapper_share['LUTLOG'] + wrapper_share['DSPs']) / 2
        wrapper_comp_share = wrapper_share['LUTLOG']  # we know we hardly use DSPs...
        wrapper_mem_share = (wrapper_share['LUTMEM'] + wrapper_share['Registers'] + wrapper_share['BRAM']) / 3
        input_data_width = op.dims.inp[-1]  # image_width
        kernel_size = np.prod(op.dims.param)
        internal_delay = 1 + input_data_width + kernel_size
        cycles_used = internal_delay + np.prod(op.dims.inp)
        # latency_ns = util_dict['latency_lim_per_tensor_cycl'] * target_hw.get_performance_dict()['fpga_clk_ns']
        latency_ns = cycles_used * target_hw.get_performance_dict()['fpga_clk_ns']
        iter_hz = 1 / (latency_ns * units.nanoU)
        offer = OperationContract(op, target_hw, self, BrickImplTypes.STREAM, iter_hz, proc_comp_share, proc_mem_share,
                                  'basis', wrapper_comp_share, wrapper_mem_share, proc_share, wrapper_share)
        return offer

    def _generate_multithreshold(self, op, target_fh, layer_name, next_op=None, next_next_op=None):
        print("[DOSA:Build:ERROR] Currently, VHDL4CNN operators can only be implemented block-wise " +
              "(i.e. use build_block). IGNORING.")
        return

    def _generate_multi_threshold_container(self, multi_threshold_op_list, multiThresholdContainer_file_path, prod_width):
        assert len(multi_threshold_op_list) == self._block_multi_threshold_used_id
        with open(os.path.join(self.my_hdl_template_folder, 'MultiThresholdOperation_template.vhd'), 'r') as in_file, \
                open(multiThresholdContainer_file_path, 'w') as out_file:
            for line in in_file.readlines():
                skip_write = False
                if 'DOSA_INSERT_GENERICS_FOR_WHEN_ELSE' in line:
                    cur_id = 0
                    tab = '  '
                    for op in multi_threshold_op_list:
                        # layer_data = op.tvm_args['vars'][0]['ref'].data.numpy()
                        layer_data = op.tvm_args['by_position'][1]['ref'].data.numpy()
                        outline = '\n' + tab + f'multi_threshold_layer{cur_id}: if USED_LAYER_ID = {cur_id} generate\n'
                        skip_write = True
                        out_file.write(outline)
                        for channel_id in range(layer_data.shape[0]):
                            local_tab = tab * 2
                            outline = local_tab + f'multi_threshold_layer{cur_id}_op{channel_id}: if USED_LAYER_CHANNEL_ID = {channel_id} generate\n'
                            out_file.write(outline)
                            paramParsing.write_multi_threshold(out_file, layer_data[channel_id].astype(int),
                                                               # 2*get_bitwidth_of_DosaDtype(op.used_dtype),
                                                               prod_width,
                                                               get_bitwidth_of_DosaDtype(op.used_dtype),
                                                               tab_factor=3)
                            outline = local_tab + f'end generate multi_threshold_layer{cur_id}_op{channel_id};\n'
                            out_file.write(outline)
                    outline = tab + f'end generate multi_threshold_layer{cur_id};\n\n'
                    out_file.write(outline)
                else:
                    outline = line
                if not skip_write:
                    out_file.write(outline)

