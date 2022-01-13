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

from dimidium.backend.buildTools.BaseBuild import BaseHwBuild
from dimidium.backend.buildTools.cFBuild1 import cFBuild1
from dimidium.backend.operatorSets.BaseOSG import BaseOSG
from dimidium.backend.devices.dosa_device import DosaHwClasses
from dimidium.lib.dosa_dtype import get_bitwidth_of_DosaDtype, DosaDtype
from dimidium.lib.util import BrickImplTypes
from dimidium.middleend.archGen.ArchBrick import ArchBrick
from dimidium.backend.operatorSets.relay_ops import op as relay_op_list
import dimidium.backend.operatorSets.lib.haddoc2.paramsParsing as paramParsing
import dimidium.backend.operatorSets.lib.haddoc2.topologyParsing as topologyParsing


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
            if 'conv1d' in e or 'conv2d' in e:
                self.relay2osg['nn'][e] = self._generate_hdl_conv_instance
            elif 'pool1d' in e or 'pool2d' in e:
                self.relay2osg['nn'][e] = self._generate_hdl_pool_instance
            elif 'tanh' in e:
               self.relay2osg['nn'][e] = self._generate_hdl_tanh_instance
            elif 'flatten' in e:
                self.relay2osg['nn'][e] = self._generate_hdl_flatten_instance
            elif 'dropout' in e:
                self.relay2osg['nn'][e] = self._generate_hdl_dropout_instance

    def _copy_hdl_lib(self, target_hdl_dir):
        os.system("cp -n {}/* {}/".format(self.my_hdl_template_folder, target_hdl_dir))

    def build_block(self, arch_block, build_tool):
        assert isinstance(build_tool, BaseHwBuild)
        if isinstance(build_tool, cFBuild1):
            # for cF, everything under hdl/ will be included
            used_dir_path = build_tool.add_ip_dir(arch_block, is_vhdl=True)
        else:
            used_dir_path = build_tool.add_ip_dir(arch_block)
        # first, copy all lib files
        # self._copy_hdl_lib(used_dir_path)
        # use global_vhdl_dir to have it only once per node
        self._copy_hdl_lib(build_tool.global_vhdl_dir)
        paramFile = os.path.abspath(used_dir_path + '/params.vhd')      # Configuration VHDL output
        topFile = os.path.abspath(used_dir_path + '/cnn_process.vhd')  # Top level VHDL output
        bitwidthFile = os.path.abspath(used_dir_path + '/bitwidths.vhd')   # Bitwidth  VHDL output

        used_dtype = DosaDtype.UNKNOWN
        layer_names_by_op_id = {}
        layer_names_ordered = []
        input_dims_by_op_id = {}
        input_dims_ordered = []
        ops_implemented_ordered = []
        wrapper_flatten_op = None
        # as haddoc, we first take care of the params
        with open(paramFile, 'w') as vhdlf:
            paramParsing.write_fileHead(vhdlf)
            # now, iterate over bricks
            for bb in arch_block.brick_list:
                #for op in bb.local_op_iter_gen():
                for op_i in bb.ops.keys():
                    op = bb.ops[op_i]
                    next_i = op_i + 1
                    next_op = None
                    if next_i in bb.ops.keys():
                        next_op = bb.ops[next_i]
                    layer_name = self._create_unique_layer_name(op.name)
                    if used_dtype == DosaDtype.UNKNOWN:
                        used_dtype = op.used_dtype
                    elif used_dtype != op.used_dtype:
                        print("[DOSA:OSG:ERROR] Haddoc supports only one bit width per block. Trying to ignore...")

                    if 'pool1d' in op.op_call or 'pool2d' in op.op_call:
                        layer_names_by_op_id[op.global_op_id] = layer_name
                        layer_names_ordered.append(layer_name)
                        input_dims_by_op_id[op.global_op_id] = op.dims.inp
                        input_dims_ordered.append(op.dims.inp)
                        ops_implemented_ordered.append(op)
                        self._param_parse_pool(op, vhdlf, layer_name)
                    elif 'conv1d' in op.op_call or 'conv2d' in op.op_call:
                        # TODO map 1d correctly
                        layer_names_by_op_id[op.global_op_id] = layer_name
                        layer_names_ordered.append(layer_name)
                        input_dims_by_op_id[op.global_op_id] = op.dims.inp
                        input_dims_ordered.append(op.dims.inp)
                        ops_implemented_ordered.append(op)
                        if next_op is not None and 'bias' not in next_op.op_call:
                            # useless -> None again
                            next_op = None
                        self._param_parse_conv(op, vhdlf, layer_name, next_op)
                    elif 'flatten' in op.op_call:
                        # layer name will be ignored
                        wrapper_flatten_op = op
                        if op_i + 1 != len(bb.ops):
                            print("[DOSA:Haddoc2OSG:ERROR] flatten is only supported as last layer!. STOP.")
                            exit(1)
                    elif 'dropout' in op.op_call:
                        print("[DOSA:OSG:ERROR] Not yet implemented!. STOP.")
                        exit(1)
                    else:
                        print("[DOSA:OSG:ERROR] Not yet implemented!. STOP.")
                        exit(1)

            paramParsing.write_fileEnd(vhdlf)
        # then, we do the bitwidth
        used_bit_width = get_bitwidth_of_DosaDtype(used_dtype)
        input_dim = input_dims_ordered[0]
        input_bit_width = used_bit_width
        # TODO: make dynamic
        input_bit_width *= input_dim[1]  # num channels
        output_dim = ops_implemented_ordered[-1].dims.out
        output_bit_width = used_bit_width
        # TODO: make dynamic
        output_bit_width *= output_dim[1]
        self._generate_bitwidth(bitwidthFile, used_bit_width, input_bit_width, output_bit_width)
        wrapper_last_op = None
        # finally, create the topology
        with open(topFile, 'w') as topf:
            topologyParsing.WriteLibs(topf)
            topologyParsing.WriteEntity(topf)
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
                if 'conv1d' in op.op_call or 'conv2d' in op.op_call:
                    topologyParsing.InstanceConvLayer(topf, layer_name, previous_layer_name)
                elif 'pool1d' in op.op_call or 'pool2d' in op.op_call:
                    topologyParsing.InstancePoolLayer(topf, layer_name, previous_layer_name)
                elif 'tanh' in op.op_call:
                    # is done implicitly, in ConvLayer
                    # TODO: way to deactivate if not tanh activation?
                    pass
                previous_layer_name = layer_name
            # TODO: other ending?
            topologyParsing.InstanceDynOutputLayer(topf, previous_layer_name)
            topologyParsing.WriteArchitectureEnd(topf)
        # TODO: generate wrapper
        # wrapper_flatten_op
        # wrapper_last_op
        # wrapper_first_op = ops_implemented_ordered[0]
        # TODO: generate vhdl component and instance, for node integration
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

    def _generate_hdl_conv_instance(self, todo):
        print("[DOSA:Build:ERROR] Currently, Haddoc2 operators can only be implemented block-wise " +
              "(i.e. use build_block). IGNORING.")
        return

    def _generate_hdl_pool_instance(self, todo):
        print("[DOSA:Build:ERROR] Currently, Haddoc2 operators can only be implemented block-wise " +
              "(i.e. use build_block). IGNORING.")
        return

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

    def _create_unique_layer_name(self, op_name):
        base_str = op_name.replace('.', '_')
        name_cnt = 1
        while base_str in self.existing_layer_names:
            base_str += "_{}".format(name_cnt)
            name_cnt += 1
        self.existing_layer_names.append(base_str)
        return base_str

    def _param_parse_conv(self, op, target_fh, layer_name, bias_op=None):
        out_channel_num = op.dims.out[1]  # out_size
        in_channel_num = op.dims.inp[1]  # previous_layer_size
        kernel_size = op.dims.param[2]
        assert kernel_size == op.dims.param[3]
        input_data_width = op.dims.inp[2]  # image_width
        assert input_data_width == op.dims.inp[4]
        assert isinstance(op.tvm_args['by_position'][1]['ref'], tvm.relay.expr.Constant)
        kernel_data = op.tvm_args['by_position'][1]['ref'].data.numpy()
        bias_data = np.zeros(out_channel_num, dtype=float)
        if bias_op is not None:
            if isinstance(bias_op.tvm_args['by_position'][1]['ref'], tvm.relay.expr.Constant):
                bias_data = bias_op.tvm_args['by_position'][1]['ref'].data.numpy()
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
        return

    def _param_parse_pool(self, op, target_fh, layer_name):
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
        return

    def _generate_bitwidth(self, bitwidth_file, general_bitwidth, input_bitwidth, output_bitwidth):
        with open(bitwidth_file, 'w') as f:
            f.write('library ieee;\n')
            f.write('  use ieee.std_logic_1164.all;\n')
            f.write('  use ieee.numeric_std.all;\n')
            f.write('  use ieee.math_real.all;\n')
            f.write('package bitwidths is\n')
            f.write('  constant GENERAL_BITWIDTH      : integer := ' + str(general_bitwidth) + ';\n')
            f.write('  constant SUM_WIDTH        : integer := 3*GENERAL_BITWIDTH;\n')
            f.write('  constant INPUT_BIT_WIDTH  : integer := ' + str(input_bitwidth) + ';\n')
            f.write('  constant OUTPUT_BITWIDTH  : integer := ' + str(output_bitwidth) + ';\n')
            f.write('end bitwidths;\n')
        return



