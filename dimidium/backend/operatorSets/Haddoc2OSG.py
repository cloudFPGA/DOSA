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

from dimidium.backend.buildTools.BaseBuild import HwBuildTopVhdl
from dimidium.backend.buildTools.cFBuild1 import cFBuild1
from dimidium.backend.codeGen.Haddoc2Wrapper import Haddoc2Wrapper
from dimidium.backend.operatorSets.BaseOSG import BaseOSG
from dimidium.backend.devices.dosa_device import DosaHwClasses
from dimidium.lib.dosa_dtype import get_bitwidth_of_DosaDtype, DosaDtype
from dimidium.lib.util import BrickImplTypes
from dimidium.middleend.archGen.ArchBrick import ArchBrick
from dimidium.backend.operatorSets.relay_ops import op as relay_op_list
import dimidium.backend.operatorSets.lib.haddoc2.paramsParsing as paramParsing
import dimidium.backend.operatorSets.lib.haddoc2.topologyParsing as topologyParsing
from dimidium.backend.codeGen.WrapperInterfaces import InterfaceAxisFifo, wrapper_default_interface_bitwidth


class Haddoc2OSG(BaseOSG):

    def __init__(self):
        super().__init__('haddoc2 OSG', [DosaHwClasses.FPGA_generic, DosaHwClasses.FPGA_xilinx], '/t/b/a',
                         [BrickImplTypes.STREAM])
        self.priority = 99
        me_abs_dir = os.path.dirname(os.path.realpath(__file__))
        self.my_hdl_template_folder = os.path.abspath(me_abs_dir + '/../third_party_libs/haddoc2/lib/hdl/')
        self.existing_layer_names = []

    def init(self, dosa_hw_classes_dict, priority_internal):
        self.priority_internal = priority_internal
        self.select_dosa_hw_types(dosa_hw_classes_dict)
        # relay2osg annotation,
        #  based on https://github.com/DreamIP/haddoc2/blob/master/lib/python/parseNetTopology.py
        for e in self.relay2osg['nn']:
            if 'conv2d' in e:
                self.relay2osg['nn'][e] = self._param_parse_conv
            elif 'bias_add' in e:
                self.relay2osg['nn'][e] = self._generate_hdl_biasAdd
            elif 'pool2d' in e:
                self.relay2osg['nn'][e] = self._param_parse_pool
            elif 'tanh' in e:
                self.relay2osg['nn'][e] = self._generate_hdl_tanh_instance
            elif 'relu' in e:
                self.relay2osg['nn'][e] = self._generate_hdl_relu
            elif 'flatten' in e:
                self.relay2osg['nn'][e] = self._generate_hdl_flatten_instance
            elif 'dropout' in e:
                self.relay2osg['nn'][e] = self._generate_hdl_dropout_instance

    def _copy_hdl_lib(self, target_hdl_dir):
        os.system("cp -n {}/* {}/".format(self.my_hdl_template_folder, target_hdl_dir))

    def build_block(self, arch_block, build_tool):
        assert isinstance(build_tool, HwBuildTopVhdl)
        if isinstance(build_tool, cFBuild1):
            # for cF, everything under hdl/ will be included
            hybrid_dir = build_tool.add_ip_dir(arch_block, hybrid=True)
            used_vhdl_dir_path = hybrid_dir[0]
            used_hls_dir_path = hybrid_dir[1]
        else:
            used_vhdl_dir_path = build_tool.add_ip_dir(arch_block)
            used_hls_dir_path = used_vhdl_dir_path
        # first, copy all lib files
        # self._copy_hdl_lib(used_dir_path)
        # use global_vhdl_dir to have it only once per node
        self._copy_hdl_lib(build_tool.global_vhdl_dir)
        # file names are in specific folder, don't need to match entity name
        paramFile = os.path.abspath(used_vhdl_dir_path + '/params.vhd')  # Configuration VHDL output
        topFile = os.path.abspath(used_vhdl_dir_path + '/cnn_process.vhd')  # Top level VHDL output
        bitwidthFile = os.path.abspath(used_vhdl_dir_path + '/bitwidths.vhd')  # Bitwidth  VHDL output

        used_dtype = DosaDtype.UNKNOWN
        layer_names_by_op_id = {}
        layer_names_ordered = []
        input_dims_by_op_id = {}
        input_dims_ordered = []
        ops_implemented_ordered = []
        wrapper_flatten_op = None
        wrapper_first_brick = None
        # as haddoc, we first take care of the params
        with open(paramFile, 'w') as vhdlf:
            paramParsing.write_fileHead(vhdlf, arch_block.block_uuid)
            # now, iterate over bricks
            for bb in arch_block.brick_list:
                if wrapper_first_brick is None:
                    wrapper_first_brick = bb
                skip_i = []
                # for op in bb.local_op_iter_gen():
                for op_i in bb.ops.keys():
                    op = bb.ops[op_i]
                    layer_name = self._create_unique_layer_name(op.name)
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
                            print("[DOSA:Haddoc2OSG:ERROR] flatten is only supported as last layer!. STOP.")
                            exit(1)
                    else:
                        osg_func = self._get_osg_func(op.op_call)
                        mod_op, consumed_opt_ops = osg_func(op, vhdlf, layer_name, next_op, next_next_op)

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
                    #         print("[DOSA:Haddoc2OSG:ERROR] flatten is only supported as last layer!. STOP.")
                    #         exit(1)
                    # elif 'dropout' in op.op_call:
                    #     print("[DOSA:OSG:ERROR] Not yet implemented!. STOP.")
                    #     exit(1)
                    # else:
                    #     print("[DOSA:OSG:ERROR] Not yet implemented!. STOP.")
                    #     exit(1)

            paramParsing.write_fileEnd(vhdlf)
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
        self._generate_bitwidth(bitwidthFile, arch_block.block_uuid, used_bit_width, input_bit_width, output_bit_width)
        wrapper_last_op = None
        # finally, create the topology
        with open(topFile, 'w') as topf:
            topologyParsing.WriteLibs(topf, arch_block.block_uuid)
            topologyParsing.WriteEntity(topf, arch_block.block_uuid)
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
                    use_relu_activation = True
                    if op.haddoc_activation == 'tanh':
                        use_relu_activation = False
                    # else: default...
                    topologyParsing.InstanceConvLayer(topf, layer_name, previous_layer_name,
                                                      use_relu_activation=use_relu_activation)
                elif 'pool2d' in op.op_call:
                    topologyParsing.InstancePoolLayer(topf, layer_name, previous_layer_name)
                else:
                    continue
                previous_layer_name = layer_name
            topologyParsing.InstanceDynOutputLayer(topf, previous_layer_name)
            topologyParsing.WriteArchitectureEnd(topf)
        # TODO: generate wrapper
        # wrapper_last_op
        wrapper_input_fifo = InterfaceAxisFifo('input_{}'.format(arch_block.block_uuid),
                                               wrapper_first_brick.input_bw_Bs, build_tool.target_device)
        if build_tool.topVhdl.next_proc_comp_cnt == 0:
            # i.e. we are connected to the input
            wrapper_input_fifo.bitwidth = wrapper_default_interface_bitwidth
        if_in_bitw = wrapper_input_fifo.get_if_bitwidth()
        wrapper_output_fifo = InterfaceAxisFifo('output_{}'.format(arch_block.block_uuid),
                                                wrapper_first_brick.input_bw_Bs, build_tool.target_device)
        if_out_bitw = wrapper_output_fifo.get_if_bitwidth()
        # if_fifo_name = wrapper_input_fifo.get_if_name()
        if_axis_tcl = wrapper_input_fifo.get_tcl_lines()
        build_tool.add_tcl_entry(if_axis_tcl)

        wrapper_first_op = ops_implemented_ordered[0]
        block_wrapper = Haddoc2Wrapper(arch_block.block_uuid, wrapper_first_op.dims.inp, wrapper_first_op.dims.out,
                                       used_bit_width, if_in_bitw, if_out_bitw, used_hls_dir_path, wrapper_flatten_op,
                                       len(ops_implemented_ordered))
        block_wrapper.generate_haddoc2_wrapper()
        wrapper_inst_tcl = block_wrapper.get_tcl_lines_wrapper_inst('IP Core to connect DOSA infrastructure with '
                                                                    'Haddoc2 Layers')
        build_tool.add_tcl_entry(wrapper_inst_tcl)
        wrapper_decl = block_wrapper.get_wrapper_vhdl_decl_lines()
        wrapper_inst_tmpl = block_wrapper.get_vhdl_inst_tmpl()

        build_tool.topVhdl.add_proc_comp_inst(wrapper_decl, wrapper_inst_tmpl, wrapper_input_fifo, wrapper_output_fifo)
        return 0

    def build_container(self, container, build_tool):
        print("[DOSA:Build:ERROR] Haddoc2 OSG was asked to build an engine container, but it can't. IGNORING.")
        return -1

    # def generate_brick(self, brick_node: ArchBrick):
    #     pass

    # def generate_bricks(self, brick_nodes: [ArchBrick]):
    #     # to generate subsequent bricks at once
    #     pass

    # def comm_wrap_brick(self, todo):
    #     pass

    def estimate_flops_brick(self, brick_node: ArchBrick):
        pass

    # def _generate_hdl_conv_instance(self, todo):
    #     print("[DOSA:Build:ERROR] Currently, Haddoc2 operators can only be implemented block-wise " +
    #           "(i.e. use build_block). IGNORING.")
    #     return

    # def _generate_hdl_pool_instance(self, todo):
    #     print("[DOSA:Build:ERROR] Currently, Haddoc2 operators can only be implemented block-wise " +
    #           "(i.e. use build_block). IGNORING.")
    #     return

    def _generate_hdl_tanh_instance(self, todo):
        # is merged with ConvLayer.vhd?
        # but separate TanH exist, so could be added
        print("[DOSA:Build:ERROR] Currently, Haddoc2 operators can only be implemented block-wise " +
              "(i.e. use build_block). IGNORING.")
        return

    def _generate_hdl_flatten_instance(self, todo):
        print("[DOSA:Build:ERROR] Currently, Haddoc2 operators can only be implemented block-wise " +
              "(i.e. use build_block). IGNORING.")
        return

    def _generate_hdl_dropout_instance(self, todo):
        print("[DOSA:Build:ERROR] Currently, Haddoc2 operators can only be implemented block-wise " +
              "(i.e. use build_block). IGNORING.")
        return

    def _generate_hdl_biasAdd(self, todo):
        print("[DOSA:Build:ERROR] Currently, Haddoc2 operators can only be implemented block-wise " +
              "(i.e. use build_block). IGNORING.")
        return

    def _generate_hdl_relu(self, todo):
        print("[DOSA:Build:ERROR] Currently, Haddoc2 operators can only be implemented block-wise " +
              "(i.e. use build_block). IGNORING.")
        return

    def _create_unique_layer_name(self, op_name):
        base_str = op_name.replace('.', '_')
        name_cnt = 1
        while base_str in self.existing_layer_names:
            base_str += "_{}".format(name_cnt)
            name_cnt += 1
        self.existing_layer_names.append(base_str)
        return base_str

    def _param_parse_conv(self, op, target_fh, layer_name, next_op=None, next_next_op=None):
        consumed_opt_ops = 0
        out_channel_num = op.dims.out[1]  # out_size
        in_channel_num = op.dims.inp[1]  # previous_layer_size
        kernel_size = op.dims.param[2]
        assert kernel_size == op.dims.param[3]
        input_data_width = op.dims.inp[2]  # image_width
        assert input_data_width == op.dims.inp[3]
        assert isinstance(op.tvm_args['by_position'][1]['ref'], tvm.relay.expr.Constant)
        kernel_data = op.tvm_args['by_position'][1]['ref'].data.numpy()
        bias_data = np.zeros(out_channel_num, dtype=float)

        if next_op is not None and next_op.op_call == 'nn.bias_add':
            if isinstance(next_op.tvm_args['by_position'][1]['ref'], tvm.relay.expr.Constant):
                bias_data = next_op.tvm_args['by_position'][1]['ref'].data.numpy()
            else:
                print("[DOSA:OSG:WARNING] Strange non-constant bias value for op {}".format(repr(next_op)))
            consumed_opt_ops += 1

        if next_next_op is not None and next_next_op.op_call == 'nn.relu':
            op.haddoc_activation = 'relu'
            consumed_opt_ops += 1
        elif next_next_op is not None and next_next_op.op_call == 'nn.tanh':
            op.haddoc_activation = 'tanh'
            consumed_opt_ops += 1
        else:
            op.haddoc_activation = 'default'

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
        return op, consumed_opt_ops

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
        return None, 0

    def _generate_bitwidth(self, block_id, bitwidth_file, general_bitwidth, input_bitwidth, output_bitwidth):
        with open(bitwidth_file, 'w') as f:
            f.write('library ieee;\n')
            f.write('  use ieee.std_logic_1164.all;\n')
            f.write('  use ieee.numeric_std.all;\n')
            f.write('  use ieee.math_real.all;\n')
            f.write('package bitwidths_b{} is\n'.format(block_id))
            f.write('  constant GENERAL_BITWIDTH      : integer := ' + str(general_bitwidth) + ';\n')
            f.write('  constant SUM_WIDTH        : integer := 3*GENERAL_BITWIDTH;\n')
            f.write('  constant INPUT_BIT_WIDTH  : integer := ' + str(input_bitwidth) + ';\n')
            f.write('  constant OUTPUT_BITWIDTH  : integer := ' + str(output_bitwidth) + ';\n')
            f.write('end bitwidths_b{};\n'.format(block_id))
        return

