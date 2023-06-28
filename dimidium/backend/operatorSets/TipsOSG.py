#  /*******************************************************************************
#   * Copyright 2019 -- 2023 IBM Corporation
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
#  *     Created: May 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        DOSA OSG to implement TIPS Engines on FPGAs
#  *
#  *
import os
from types import SimpleNamespace

import numpy as np
import tvm
import tvm.relay as relay
import math
import json

import dimidium.lib.singleton as dosa_singleton
from dimidium.backend.buildTools.BaseBuild import HwBuildTopVhdl
from dimidium.backend.buildTools.cFBuild1 import cFBuild1
from dimidium.backend.operatorSets.BaseOSG import BaseOSG
from dimidium.backend.devices.dosa_device import DosaHwClasses
from dimidium.lib.dosa_dtype import get_bitwidth_of_DosaDtype, DosaDtype, complete_dtype_list
from dimidium.lib.util import BrickImplTypes
from dimidium.middleend.archGen.ArchBrick import ArchBrick
from dimidium.backend.operatorSets.relay_ops import op as relay_op_list
from dimidium.backend.codeGen.WrapperInterfaces import InterfaceAxisFifo, wrapper_default_interface_bitwidth
from dimidium.backend.codeGen.TipsCore import TipsCore
from dimidium.middleend.archGen.OperationContract import OperationContract
from dimidium.backend.operatorSets.lib.util import get_avg_util_dict_bytes_based, get_share_of_FPGA_resources
import dimidium.lib.units as units

__filedir__ = os.path.dirname(os.path.abspath(__file__))
__db_path__ = __filedir__ + '/osg_impl_db.json'


def _get_twoscomplement_hex_string(x, nbits):
    return hex(int(np.binary_repr(int(x), width=nbits), 2))


class TipsOSG(BaseOSG):

    def __init__(self):
        super().__init__('Tips OSG', [DosaHwClasses.FPGA_xilinx, DosaHwClasses.FPGA_generic],
                         [DosaDtype.int8, DosaDtype.uint8, DosaDtype.int16, DosaDtype.int32],
                         [BrickImplTypes.ENGINE])
        self.priority = 99
        me_abs_dir = os.path.dirname(os.path.realpath(__file__))
        self.my_template_dir = os.path.abspath(me_abs_dir + '/lib/tips/')
        self.util_db = {}
        self.avg_util_dict = {}
        self.pipeline_tensor_store = 1
        self.supports_op_padding = True
        self.op0_list = []
        self.op1_list = []
        self.hls_params = SimpleNamespace()
        self.hls_params.prog_tmpl = '{{ .opcode = {opc}, .op_param = {opp},\n' + \
                                    '  .in_addr = {in_a}, .in_length = {in_l},\n' + \
                                    '  .op0_addr = {op0_a}, .op0_length = {op0_l},\n' + \
                                    '  .op1_addr = {op1_a}, .op1_length = {op1_l},\n' + \
                                    '  .out_addr = {out_a}, .out_length = {out_l}\n' + \
                                    '}}'
        self.hls_params.network_alias_addr = 'NETWORK_ALIAS_ADDRESS'
        self.hls_params.accum_alias_addr = 'ACCUM_ALIAS_ADDRESS'
        self.hls_params.no_addr_alias = 'NO_ADDRESS_ALIAS'
        self._general_max_input_func = lambda op: (2048*8)/get_bitwidth_of_DosaDtype(op.used_dtype)
        self.base_dict = {}

    def _init_util_db_(self):
        with open(__db_path__, 'r') as infile:
            util_data = json.load(infile)
        my_util = util_data[self.name]
        self.util_db = {}
        compl_list = []
        for e in my_util:
            if 'IGNORE' in e['comment']:
                continue
            if e['device'] not in self.util_db:
                self.util_db[e['device']] = [e]
            else:
                self.util_db[e['device']].append(e)
            compl_list.append(e)
        self.avg_util_dict = get_avg_util_dict_bytes_based(compl_list, consider_paramB=True, consider_ops_num=True,
                                                           always_consider_input=False)
        bytes_total = 64 # as base
        self.base_dict['LUTLOG'] = self.avg_util_dict['LUTLOG'] * bytes_total
        self.base_dict['LUTMEM'] = self.avg_util_dict['LUTMEM'] * bytes_total
        self.base_dict['Registers'] = self.avg_util_dict['Registers'] * bytes_total
        self.base_dict['BRAM'] = self.avg_util_dict['BRAM'] * bytes_total
        self.base_dict['DSPs'] = self.avg_util_dict['DSPs'] * bytes_total
        # self.base_dict['latency_lim_per_tensor_cycl'] = self.avg_util_dict['latency_lim_per_tensor_cycl'] * bytes_total

    def _get_impl_prediction(self, op, target_hw, impl_type, consider_paramB=False, fallback_ops=None,
                             custom_byte_factor=1.0, custom_latency=None, max_param_dim=-1, max_input_dim=-1):
        # if impl_type != BrickImplTypes.ENGINE or \
        #         (target_hw.hw_class != DosaHwClasses.FPGA_xilinx and target_hw.hw_class != DosaHwClasses.FPGA_generic):
        #     return None
        if max_param_dim > 0:
            op_param_dim = 1
            for d in op.dims.param:
                op_param_dim *= d
            if op_param_dim > max_param_dim:
                print("[DOSA:TIPS:INFO] Can't offer an implementation for {}, due to exceeded parameter size."
                      .format(repr(op)))
                return None
        if max_input_dim < 0:
            max_input_dim = self._general_max_input_func(op)
        op_input_dim = np.prod(op.dims.inp)
        if op_input_dim > max_input_dim:
            print("[DOSA:TIPS:INFO] Can't offer an implementation for {}, due to exceeded input size."
                  .format(repr(op)))
            return None
        op_str = op.op_call.split('.')[-1]
        dtype_str = 'int8'  # default?
        if op.used_dtype != DosaDtype.UNKNOWN:
            dtype_str = repr(op.used_dtype)
        relevant_entries = []
        exact_matches = []
        fallback_entries = []
        # TODO: prefer entries with shorter ops list?
        for dk in self.util_db:
            if dk == target_hw.type_str:
                for e in self.util_db[dk]:
                    if op_str in e['ops']:
                        relevant_entries.append(e)
                        if e['latency_lim_per_tensor_cycl'] > 0:
                            if consider_paramB:
                                if e['inpB'] == op.input_bytes and e['paramB'] == op.parameter_bytes:
                                    exact_matches.append(e)
                            else:
                                if e['inpB'] == op.input_bytes:
                                    exact_matches.append(e)
                    if fallback_ops is not None:
                        for f in fallback_ops:
                            if f in e['ops']:
                                fallback_entries.append(e)
                                break
        res_dict = {}
        used_fallback = False
        if len(relevant_entries) == 0:
            res_dict = self.avg_util_dict
            used_fallback = True
        elif len(exact_matches) > 0:
            res_dict = get_avg_util_dict_bytes_based(exact_matches, consider_paramB=consider_paramB,
                                                     consider_ops_num=True, always_consider_input=False)
        else:
            if len(relevant_entries) == 0:
                relevant_entries = fallback_entries
                used_fallback = True
            res_dict = get_avg_util_dict_bytes_based(relevant_entries, consider_paramB=consider_paramB,
                                                     consider_ops_num=True, always_consider_input=False)
        util_dict = {}
        bytes_total = op.input_bytes
        if consider_paramB:
            # bytes_total += op.parameter_bytes
            bytes_total = op.parameter_bytes
        bytes_total *= custom_byte_factor
        # util_dict['LUTLOG'] = res_dict['LUTLOG'] * bytes_total
        # util_dict['LUTMEM'] = res_dict['LUTMEM'] * bytes_total
        # util_dict['Registers'] = res_dict['Registers'] * bytes_total
        # util_dict['BRAM'] = res_dict['BRAM'] * bytes_total
        # util_dict['DSPs'] = res_dict['DSPs'] * bytes_total
        # substract the basis, but not getting negative
        util_dict['LUTLOG'] = max(res_dict['LUTLOG'] * bytes_total - self.base_dict['LUTLOG'], 0)
        util_dict['LUTMEM'] = max(res_dict['LUTMEM'] * bytes_total - self.base_dict['LUTMEM'], 0)
        util_dict['Registers'] = max(res_dict['Registers'] * bytes_total - self.base_dict['Registers'], 0)
        util_dict['BRAM'] = max(res_dict['BRAM'] * bytes_total - self.base_dict['BRAM'], 0)
        util_dict['DSPs'] = max(res_dict['DSPs'] * bytes_total - self.base_dict['DSPs'], 0)
        util_dict['latency_lim_per_tensor_cycl'] = res_dict['latency_lim_per_tensor_cycl'] \
                                                   * op.input_bytes
                                                    # * op.parameter_bytes
        # * (op.input_bytes + op.parameter_bytes)
        wrapper_dict = {'LUTLOG': 0.0, 'LUTMEM': 0.0, 'Registers': 0.0, 'BRAM': 0.0, 'DSPs': 0.0}

        fpga_utility = target_hw.get_resource_dict()['FPGA_utility']
        proc_share = get_share_of_FPGA_resources(fpga_utility, util_dict)
        # wrapper_share = get_share_of_FPGA_resources(fpga_utility, wrapper_dict)
        wrapper_share = wrapper_dict
        # proc_comp_share = (proc_share['LUTLOG'] + proc_share['DSPs']) / 2
        proc_comp_share = proc_share['LUTLOG']  # we know we hardly use DSPs..
        # proc_mem_share = (proc_share['LUTMEM'] + proc_share['Registers'] + proc_share['BRAM']) / 3
        proc_mem_share = max(proc_share['LUTMEM'], proc_share['Registers'], proc_share['BRAM'])
        # wrapper_comp_share = (wrapper_share['LUTLOG'] + wrapper_share['DSPs']) / 2
        # wrapper_comp_share = wrapper_share['LUTLOG']  # we know we hardly use DSPs...
        wrapper_comp_share = 0
        # wrapper_mem_share = (wrapper_share['LUTMEM'] + wrapper_share['Registers'] + wrapper_share['BRAM']) / 3
        # wrapper_mem_share = max(wrapper_share['LUTMEM'], wrapper_share['Registers'], wrapper_share['BRAM'])
        wrapper_mem_share = 0
        base_share = get_share_of_FPGA_resources(fpga_utility, self.base_dict)
        base_comp_share = base_share['LUTLOG']  # we know we hardly use DSPs..
        base_mem_share = max(base_share['LUTMEM'], base_share['Registers'], base_share['BRAM'])
        limit_bw_Bs = target_hw.get_performance_dict()['bw_dram_gBs'] * units.gigaU

        if custom_latency is None:
            latency_ns = util_dict['latency_lim_per_tensor_cycl'] * target_hw.get_performance_dict()['fpga_clk_ns']
            iter_hz = 1 / (latency_ns * units.nanoU)
        else:
            latency_ns = custom_latency * target_hw.get_performance_dict()['fpga_clk_ns']
            iter_hz = 1 / (latency_ns * units.nanoU)
        adapted_peak_performance = op.flops * iter_hz

        offer = OperationContract(op, target_hw, self, BrickImplTypes.ENGINE, iter_hz, proc_comp_share, proc_mem_share,
                                  'default', wrapper_comp_share, wrapper_mem_share, proc_share, wrapper_share,
                                  base_comp_share, base_mem_share, base_share, adapted_peak_performance, limit_bw_Bs)
        return offer

    def _get_dyn_costs(self, contract, add_brick, target_hw):
        pseudo_brick = ArchBrick()
        pseudo_brick.used_dtype = contract.brick.used_dtype
        op_list = list(contract.brick.ops.values())
        op_list.extend(list(add_brick.ops.values()))
        pseudo_brick.reconstruct_from_op_list(op_list)
        self.annotate_brick(pseudo_brick, contract.device)
        new_contract = pseudo_brick.available_contracts[0]
        comp_costs = (new_contract.comp_util_share - contract.comp_util_share)
        mem_costs = (new_contract.mem_util_share - contract.mem_util_share)
        return comp_costs, mem_costs, new_contract.iter_hz

    def init(self, dosa_hw_classes_dict, priority_internal):
        self.priority_internal = priority_internal
        self.select_dosa_hw_types(dosa_hw_classes_dict)
        self._init_util_db_()
        for e in self.relay2osg['nn']:
            if 'dense' in e:
                self.relay2osg['nn'][e] = self._parse_dense, \
                                          lambda op, thw, it: self._get_impl_prediction(op, thw, it,
                                                                                        consider_paramB=True,
                                                                                        custom_byte_factor=0.0015,
                                                                                        # max_param_dim=1024
                                                                                        max_param_dim=32768
                                                                                        )
                self.op0_list.append(e)
            elif 'bias_add' in e:
                self.relay2osg['nn'][e] = self._parse_biasAdd, \
                                          lambda op, thw, it: self._get_impl_prediction(op, thw, it,
                                                                                        consider_paramB=True,
                                                                                        fallback_ops=['add'],
                                                                                        custom_byte_factor=0.02,
                                                                                        custom_latency=10)
                self.op1_list.append(e)
            elif 'tanh' in e:
                self.relay2osg['nn'][e] = self._parse_tanh, \
                                          lambda op, thw, it: self._get_impl_prediction(op, thw, it,
                                                                                        consider_paramB=False)
                self.op0_list.append(e)
            elif 'relu' in e:
                self.relay2osg['nn'][e] = self._parse_relu, \
                                          lambda op, thw, it: self._get_impl_prediction(op, thw, it,
                                                                                        consider_paramB=False,
                                                                                        custom_latency=10)
                self.op0_list.append(e)
            elif 'flatten' in e:
                self.relay2osg['nn'][e] = self._parse_flatten, \
                                          lambda op, thw, it: self._get_impl_prediction(op, thw, it,
                                                                                        consider_paramB=False,
                                                                                        custom_byte_factor=0,
                                                                                        custom_latency=1)
        for e in self.relay2osg:
            if type(self.relay2osg[e]) == dict:
                continue
            elif 'add' in e:
                self.relay2osg[e] = self._parse_biasAdd, \
                                    lambda op, thw, it: self._get_impl_prediction(op, thw, it,
                                                                                  consider_paramB=True,
                                                                                  custom_byte_factor=0.02,
                                                                                  custom_latency=10)
                self.op1_list.append(e)
            elif 'tanh' in e:
                self.relay2osg[e] = self._parse_tanh, \
                                    lambda op, thw, it: self._get_impl_prediction(op, thw, it,
                                                                                  consider_paramB=False)
                self.op0_list.append(e)

    def build_block(self, arch_block, build_tool, selected_contracts):
        print("[DOSA:Build:ERROR] TIPS OSG was asked to build a streaming block, but it can't. IGNORING.")
        return -1

    def build_container(self, container, build_tool, selected_contracts):
        assert isinstance(build_tool, HwBuildTopVhdl)
        arch_block = container.block_ref
        used_dir_path = build_tool.add_ip_dir(arch_block.block_uuid)
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
        fractional_bits = 0
        accum_bitw = cur_w
        if used_dtype == DosaDtype.float16 or used_dtype == DosaDtype.float32:
            precision_string = 'ap_fixed<16,6>'  # TODO
            accum_string = precision_string
            fractional_bits = 10
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
                accum_bitw = cur_w * accum_factor
            else:
                accum_string = precision_string

        # find largest dimensions
        largest_input = 0
        largest_op0 = 0
        largest_op1 = 0
        largest_output = 0
        for bb in arch_block.brick_list:
            for op in bb.local_op_iter_gen():
                op_i = np.prod(op.dims.inp)
                if op_i > largest_input:
                    largest_input = op_i
                op_o = np.prod(op.dims.out)
                if op_o > largest_output:
                    largest_output = op_o
                op_op = np.prod(op.dims.param)
                op_str = op.op_call.split('.')[-1]
                if op_str in self.op0_list and op_op > largest_op0:
                    largest_op0 = op_op
                if op_str in self.op1_list and op_op > largest_op1:
                    largest_op1 = op_op

        # TODO: later
        #   iterate over container.ops and decide for op_codes dynamically
        #   maybe also use to decide largest_op0... parameter
        opcodes = {'nop': ['TIPS_NOP'], 'dense': ['DENSE', 'DENSE_BIAS'], 'relu': ['RELU'], 'tanh': ['TANH']}

        # implement
        op_rom = []
        program = []
        cur_addr = 0
        contr_i = 0
        bb_i = 0
        wrapper_first_brick = None
        wrapper_last_brick = None
        for bb in arch_block.brick_list:
            skip_i = []
            if wrapper_first_brick is None:
                wrapper_first_brick = bb
            wrapper_last_brick = bb
            cur_brick_contr = selected_contracts[contr_i]
            for op_i in bb.ops.keys():
                if op_i in skip_i:
                    continue
                op = bb.ops[op_i]
                op_c = cur_brick_contr.get_contract_to_op(op)
                op_str = op.op_call.split('.')[-1]
                op_opcode = opcodes[op_str]
                osg_func = self._get_osg_func(op.op_call)

                first_instr = False
                last_instr = False
                if bb_i == 0 and op_i == 0:
                    first_instr = True
                elif bb_i == (len(arch_block.brick_list) - 1) and op_i == (len(bb.ops.keys()) - 1):
                    last_instr = True
                next_i = op_i + 1
                next_op = None
                if next_i in bb.ops.keys():
                    next_op = bb.ops[next_i]

                prog, op_r, consumed_next = osg_func(op, op_opcode, first_instr, last_instr, used_dtype, largest_input,
                                                     largest_op0, largest_op1, largest_output, cur_addr,
                                                     next_op=next_op)
                if consumed_next:
                    skip_i.append(next_i)
                if prog is not None:
                    program.append(prog)
                if len(op_r) > 0:
                    # op_rom.extend(op_r)
                    op_rom.append(op_r)  # better append to be able to add comments?
                    cur_addr += len(op_r)
            contr_i += 1
            bb_i += 1

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

        tips_inst = TipsCore(arch_block.block_uuid, cur_w, if_in_bitw, if_out_bitw, used_dir_path, program, op_rom,
                             largest_input, largest_op0, largest_op1, largest_output,
                             precision_string, accum_string, fractional_bits, accum_bitw)
        tips_inst.generate()

        build_tool.add_makefile_entry(used_dir_path, 'all')
        tips_inst_tcl = tips_inst.get_tcl_lines_inst()
        build_tool.add_tcl_entry(tips_inst_tcl)
        tips_inst_decl = tips_inst.get_vhdl_decl_lines()
        tips_inst_inst_tmpl = tips_inst.get_vhdl_inst_tmpl()
        build_tool.topVhdl.add_proc_comp_inst(arch_block, tips_inst_decl, tips_inst_inst_tmpl, wrapper_input_fifo,
                                              wrapper_output_fifo)

        # adding debug
        tcl_tmp, decl_tmp, inst_tmp = wrapper_input_fifo.get_debug_lines()
        build_tool.topVhdl.debug_core.add_new_probes(tcl_tmp, decl_tmp, inst_tmp)
        # unsure if output will be used --> add debug lines later
        tcl_tmp, decl_tmp, inst_tmp = tips_inst.get_debug_lines()
        build_tool.topVhdl.debug_core.add_new_probes(tcl_tmp, decl_tmp, inst_tmp)
        return 0

    def _parse_dense(self, op, opcode, first: bool, last: bool, used_ram_dtype, longest_input, longest_op0, longest_op1,
                     longest_output, cur_addr, next_op=None):
        opc = opcode[0]
        bias_data = None
        orig_data = op.tvm_args['by_position'][1]['ref'].data.numpy()
        consumed_next_op = False
        in_a = self.hls_params.accum_alias_addr
        if first:
            in_a = self.hls_params.network_alias_addr
        out_a = self.hls_params.accum_alias_addr
        if last:
            out_a = self.hls_params.network_alias_addr
        # for now...TODO: later allow casting
        assert used_ram_dtype == op.used_dtype
        assert op.dims.inp[0] == 1  # batch_size 1 for now
        assert op.dims.inp[0] == op.dims.out[0]
        in_l = np.prod(op.dims.inp)
        out_l = np.prod(op.dims.out)
        opp = in_l
        op0_a = cur_addr
        op0_l = np.prod(op.dims.param)
        op1_a = self.hls_params.no_addr_alias
        op1_l = 0
        nbits = get_bitwidth_of_DosaDtype(used_ram_dtype)

        if next_op is not None and (next_op.op_call == 'add' or next_op.op_call == 'nn.bias_add'):
            opc = opcode[1]
            consumed_next_op = True
            if next_op.op_call == 'add':
                # it is still called bias
                bias_data = next_op.tvm_args['by_position'][1]['ref'].data.numpy()
            else:
                bias_data = next_op.tvm_args['by_position'][1]['ref'].data.numpy()
            op1_a = cur_addr + op0_l
            op1_l = np.prod(next_op.dims.param)
        data_string = []
        padding_cnt = 0
        for r in orig_data:
            for e in r:
                ds = _get_twoscomplement_hex_string(e, nbits)
                data_string.append(ds)
            # padding
            # longest input = row-size of dense
            # also necessary after last line
            if len(r) < longest_input:
                for i in range(len(r), longest_input):
                    data_string.append('0x00')
                    padding_cnt += 1
        if bias_data is not None:
            for e in bias_data:
                ds = _get_twoscomplement_hex_string(e, nbits)
                data_string.append(ds)
            # TODO: no padding necessary?

        if op1_a != self.hls_params.no_addr_alias:
            op1_a += padding_cnt
        prog = self.hls_params.prog_tmpl.format(opc=opc, opp=opp, in_a=in_a, in_l=in_l, op0_a=op0_a, op0_l=op0_l,
                                                op1_a=op1_a, op1_l=op1_l, out_a=out_a, out_l=out_l)
        assert len(data_string) == (op0_l + op1_l + padding_cnt)
        assert op0_l <= longest_op0
        assert op1_l <= longest_op1
        assert op1_l <= longest_output
        return prog, data_string, consumed_next_op

    def _parse_biasAdd(self, op, opcode, first: bool, last: bool, used_ram_dtype, longest_input, longest_op0,
                       longest_op1,
                       longest_output, cur_addr, next_op=None):
        print('[DOSA:TipsOSG:ERROR] bias_add without dense is not yet supported. STOP. ')
        exit(1)
        return -1

    def _parse_tanh(self, op, opcode, first: bool, last: bool, used_ram_dtype, longest_input, longest_op0, longest_op1,
                    longest_output, cur_addr, next_op=None):
        opc = opcode[0]
        in_a = self.hls_params.accum_alias_addr
        if first:
            in_a = self.hls_params.network_alias_addr
        out_a = self.hls_params.accum_alias_addr
        if last:
            out_a = self.hls_params.network_alias_addr
        # for now...TODO: later allow casting
        assert used_ram_dtype == op.used_dtype
        assert op.dims.inp[0] == 1  # batch_size 1 for now
        assert op.dims.inp[0] == op.dims.out[0]
        in_l = np.prod(op.dims.inp)
        out_l = np.prod(op.dims.out)
        assert in_l == out_l
        opp = 0
        op0_a = self.hls_params.no_addr_alias
        op0_l = 0
        op1_a = self.hls_params.no_addr_alias
        op1_l = 0
        nbits = get_bitwidth_of_DosaDtype(used_ram_dtype)

        prog = self.hls_params.prog_tmpl.format(opc=opc, opp=opp, in_a=in_a, in_l=in_l, op0_a=op0_a, op0_l=op0_l,
                                                op1_a=op1_a, op1_l=op1_l, out_a=out_a, out_l=out_l)
        return prog, [], False

    def _parse_relu(self, op, opcode, first: bool, last: bool, used_ram_dtype, longest_input, longest_op0, longest_op1,
                    longest_output, cur_addr, next_op=None):
        opc = opcode[0]
        in_a = self.hls_params.accum_alias_addr
        if first:
            in_a = self.hls_params.network_alias_addr
        out_a = self.hls_params.accum_alias_addr
        if last:
            out_a = self.hls_params.network_alias_addr
        # for now...TODO: later allow casting
        assert used_ram_dtype == op.used_dtype
        assert op.dims.inp[0] == 1  # batch_size 1 for now
        assert op.dims.inp[0] == op.dims.out[0]
        in_l = np.prod(op.dims.inp)
        out_l = np.prod(op.dims.out)
        assert in_l == out_l
        opp = 0
        op0_a = self.hls_params.no_addr_alias
        op0_l = 0
        op1_a = self.hls_params.no_addr_alias
        op1_l = 0
        nbits = get_bitwidth_of_DosaDtype(used_ram_dtype)

        prog = self.hls_params.prog_tmpl.format(opc=opc, opp=opp, in_a=in_a, in_l=in_l, op0_a=op0_a, op0_l=op0_l,
                                                op1_a=op1_a, op1_l=op1_l, out_a=out_a, out_l=out_l)
        return prog, [], False

    def _parse_flatten(self, op, opcode, first: bool, last: bool, used_ram_dtype, longest_input, longest_op0,
                       longest_op1,
                       longest_output, cur_addr, next_op=None):
        # flatten is done anyhow, not even a NOP necessary
        return None, [], False
