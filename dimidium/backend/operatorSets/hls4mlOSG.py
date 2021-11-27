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
from dimidium.backend.buildTools.BaseBuild import BaseHwBuild
from dimidium.backend.operatorSets.BaseOSG import BaseOSG
from dimidium.backend.devices.dosa_device import DosaHwClasses
from dimidium.middleend.archGen.ArchBrick import ArchBrick
from dimidium.lib.util import BrickImplTypes
from dimidium.backend.operatorSets.relay_ops import op as relay_op_list
from dimidium.lib.dosa_dtype import get_bitwidth_of_DosaDtype, DosaDtype


class Hls4mlOSG(BaseOSG):

    def __init__(self):
        super().__init__('hls4ml OSG', [DosaHwClasses.FPGA_xilinx], '/t/b/a',
                         [BrickImplTypes.STREAM, BrickImplTypes.ENGINE])
        # no DosaHwClasses.FPGA_generic, since it is bound to xilinx?
        self.priority = 92

    def init(self, dosa_hw_classes_dict, priority_internal):
        self.priority_internal = priority_internal
        self.select_dosa_hw_types(dosa_hw_classes_dict)
        # relay2osg annotation,
        #  based on https://github.com/fastmachinelearning/hls4ml/blob/master/hls4ml/model/hls_layers.py
        #  and https://github.com/fastmachinelearning/hls4ml/tree/master/hls4ml/converters/
        for e in self.relay2osg['nn']:
            if 'conv1d' in e:
                self.relay2osg['nn'][e] = self._generate_hls_conv1d
            elif 'conv2d' in e:
                self.relay2osg['nn'][e] = self._generate_hls_conv2d
            elif 'global' in e and 'pool1d' in e:
                self.relay2osg['nn'][e] = self._generate_hls_globalPool1d
            elif 'global' in e and 'pool2d' in e:
                self.relay2osg['nn'][e] = self._generate_hls_globalPool2d
            elif 'pool1d' in e:
                self.relay2osg['nn'][e] = self._generate_hls_pool1d
            elif 'pool2d' in e:
                self.relay2osg['nn'][e] = self._generate_hls_pool2d
            elif 'prelu' in e:
                self.relay2osg['nn'][e] = self._generatae_hls_prelu
            elif 'relu' in e:
                self.relay2osg['nn'][e] = self._generate_hls_parAct
            elif 'softmax' in e:
                self.relay2osg['nn'][e] = self._generate_hls_softmax
            elif 'dense' in e:
                self.relay2osg['nn'][e] = self._generate_hls_dense
            elif 'batch_norm' in e:
                self.relay2osg['nn'][e] = self._generate_hls_batchNorm
            elif 'pad' in e:
                self.relay2osg['nn'][e] = self._generate_hls_padding
            elif 'bias_add' in e:
                self.relay2osg['nn'][e] = self._generate_hls_biasAdd
            elif 'flatten' in e or 'dropout' in e:
                self.relay2osg['nn'][e] = self._generate_hls_skipLayer
        for e in self.relay2osg:
            if type(e) == dict:
                continue
            if ('tan' in e or 'sin' in e or 'cos' in e) and 'is' not in e:
                self.relay2osg[e] = self._generate_hls_act
            elif 'add' in e or 'sub' in e or 'mul' in e or 'avg' in e \
                    or 'max' in e or 'min' in e or 'concat' in e or 'sum' in e:
                self.relay2osg[e] = self._generate_hls_merge
            elif 'transpose' in e:
                self.relay2osg[e] = self._generate_hls_transpose
            elif 'reshape' in e:
                self.relay2osg[e] = self._generate_hls_reshape
        # not covered hls4ml classes:
        #  GarNet, Resize, SeparableConv2D, DepthwiseConv2D

    def build_block(self, arch_block, build_tool):
        assert isinstance(build_tool, BaseHwBuild)
        used_dir_path = build_tool.add_ip_dir(arch_block)
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
        if used_dtype == DosaDtype.float16 or used_dtype.float32:
            precision_string = 'ap_fixed<16,6>'  # TODO
        else:
            precision_string = 'ap_uint<{}>'.format(cur_w)
        hls_config = {'Model': {'Precision': precision_string, 'ReuseFactor': '1'}}
        hls_model_config = {'OutputDir': used_dir_path, 'ProjectName': project_name, 'Backend': 'Vivado',
                            'XilinxPart': build_tool.target_device.part_string, 'Board': None,
                            'ClockPeriod': build_tool.target_device.clock_period,
                            'IOType': 'io_stream',  # or io_parallel
                            'HLSConfig':  hls_config}  # ,
                            # 'KerasJson': 'KERAS_3layer.json', 'KerasH5': 'KERAS_3layer_weights.h5'}  # TODO

        for bb in arch_block.brick_list:
            for op in bb.local_op_iter_gen():
                print(op.op_call)

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

    def _generate_hls_conv1d(self, todo):
        return

    def _generate_hls_conv2d(self, todo):
        return

    def _generate_hls_pool1d(self, todo):
        return

    def _generate_hls_pool2d(self, todo):
        return

    def _generate_hls_globalPool1d(self, todo):
        return

    def _generate_hls_globalPool2d(self, todo):
        return

    def _generatae_hls_prelu(self, todo):
        return

    def _generate_hls_parAct(self, todo):
        return

    def _generate_hls_softmax(self, todo):
        return

    def _generate_hls_dense(self, todo):
        return

    def _generate_hls_batchNorm(self, todo):
        return

    def _generate_hls_padding(self, todo):
        # including ZeroPadding
        return

    def _generate_hls_biasAdd(self, todo):
        return

    def _generate_hls_act(self, todo):
        return

    def _generate_hls_merge(self, todo):
        return

    def _generate_hls_transpose(self, todo):
        return

    def _generate_hls_reshape(self, todo):
        return

    def _generate_hls_skipLayer(self, todo):
        # see e.g. https://github.com/fastmachinelearning/hls4ml/blob/e804cfc6bbadd9b64857e2dbd2459a5b7200ffb7/hls4ml/converters/onnx_to_hls.py#L286
        return


class OsgDataReader(object):
    def __init__(self, config, data):
        self.config = config

    def _find_data(self, layer_name, var_name):
        # TODO
        return

    def get_weights_data(self, layer_name, var_name):
        data = self._find_data(layer_name, var_name)
        if data:
            return data[()]
        else:
            return None

    def get_weights_shape(self, layer_name, var_name):
        data = self._find_data(layer_name, var_name)
        if data is not None:
            return data.shape
        else:
            return None

