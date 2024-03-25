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
#  *     Created: Aug 2023
#  *     Authors: NGL
#  *
#  *     Description:
#  *        module to translate a torch script module to a brevitas module
#  *
#  *

import torch.jit
from brevitas import nn as qnn
from torch import nn
# from distutils.util import strtobool

from gradatim.dnn_quant.models.quantized import GenericQuantModule
from gradatim.dnn_quant.quantizers import *
from gradatim.dnn_quant.utils import Reshape


bitwidth_to_brevitas_defs = {
    8: {
        'act_quant': Int8ActPerTensorFixedPoint,
        'weight_quant': Int8WeightPerTensorFixedPoint,
        'bias_quant': Int8Bias,
        'bit_width': 8
    },
    5: {
        'act_quant': Int5ActPerTensorFixedPoint,
        'weight_quant': Int5WeightPerTensorFixedPoint,
        'bias_quant': Int5Bias,
        'bit_width': 5
    },
    4: {
        'act_quant': Int4ActPerTensorFixedPoint,
        'weight_quant': Int4WeightPerTensorFixedPoint,
        'bias_quant': Int4Bias,
        'bit_width': 4
    },
    3: {
        'act_quant': Int3ActPerTensorFixedPoint,
        'weight_quant': Int3WeightPerTensorFixedPoint,
        'bias_quant': Int3Bias,
        'bit_width': 3
    },
    16: {
        'act_quant': Int16ActPerTensorFloat,
        'weight_quant': Int16WeightPerTensorFloat,
        'bias_quant': Int16Bias,
        'bit_width': 16
    }
}
supported_bitwidths = sorted([k for k in bitwidth_to_brevitas_defs.keys()])


def translate_to_quantized_model(fp_model, bit_width):
    # TODO:
    #  1. count activations, weights and biases...for internal lists
    #  2. then, append layer by layer (_append())
    original_name = fp_model.original_name
    layer_dict = {}
    for name, module in fp_model.named_modules():
        if name == '':
            # is the complete module?
            continue
        if name not in layer_dict:
            layer_dict[name] = module
    forward_calls = []
    irdef_to_name_dict = {}
    for node in fp_model.graph.nodes():
        node_ir = str(node)
        if 'prim::GetAttr' in node_ir:
            defname = node_ir.split(":")[0][:-1]
            modulename = node_ir.split('prim::GetAttr[name="')[1].split('"')[0]
            irdef_to_name_dict[defname] = modulename
        if 'name="forward"' in node_ir:
            called_def = node_ir.split('name="forward"](')[1].split(",")[0]
            forward_calls.append(called_def)

    module_generator = QuantModuleGenerator(layer_dict, irdef_to_name_dict, forward_calls, bit_width, original_name)
    q_model = module_generator.generate()

    return q_model


class QuantModuleGenerator:

    def __init__(self, layer_dict, irdef_to_name_dict, forward_calls, bitwidth, model_name):
        self.layer_dict = layer_dict
        self.irdef_to_name_dict = irdef_to_name_dict
        self.forward_calls = forward_calls
        self.bitwidth = bitwidth
        self.model_name = model_name
        self._method_cache = None

    def generate(self) -> GenericQuantModule:
        self._forward_layers = []
        for call in self.forward_calls:
            self._forward_layers.append(self.layer_dict[self.irdef_to_name_dict[call]])
        num_act, num_weights, num_bias = self._count_act_weight_bias()

        q_module = GenericQuantModule(num_act, num_weights, num_bias, self.model_name)

        self._a_quant, self._w_quant, self._b_quant, self._bit_width, self._return_qt, self._do_quantization = \
            q_module._process_quant_methods(**bitwidth_to_brevitas_defs[self.bitwidth])

        self._act_cnt = 0
        self._weight_cnt = 0
        self._bias_cnt = 0

        needs_identity = True
        self._last_layer_name = ''
        self._last_layer_shape = []
        for layer in self._forward_layers:
            ln = layer.original_name
            # before actual layer
            if needs_identity:
                q_module.append(qnn.QuantIdentity(act_quant=self._a_quant[self._act_cnt],
                                                  return_quant_tensor=self._return_qt))
                q_module.append(nn.Dropout(p=GenericQuantModule.dropout))
                self._act_cnt += 1
            # actual layer:
            needs_identity = self.translate(layer, q_module)
            self._last_layer_name = ln
            if hasattr(layer, 'weight'):
                self._last_layer_shape = tuple(layer.weight.shape)

        return q_module

    def _count_act_weight_bias(self):
        num_act = 0
        num_weights = 0
        num_bias = 0
        for layer in self._forward_layers:
            ln = layer.original_name
            num_act += 1
            if ln in ['Linear', 'Conv2d']:
                num_weights += 1
                if hasattr(layer, 'bias'):
                    num_bias += 1
        return num_act, num_weights, num_bias

    def translate(self, fp_layer: torch.jit.RecursiveScriptModule, q_module: GenericQuantModule) -> bool:
        if self._method_cache is None:
            self._method_cache = {}

        method_name = fp_layer.original_name
        visitor = self._method_cache.get(method_name, None)
        if visitor is None:
            method = 'translate_' + method_name
            # get method or default
            visitor = getattr(self, method, self._abort_translation)
            self._method_cache[method_name] = visitor

        return visitor(fp_layer, q_module)

    def _abort_translation(self, fp_layer, q_module):
        print(f"[DOSA:quantization_init:ERROR] operation {fp_layer.original_name} can not be translated to a "
              f"quantized equivalent (maybe not yet implemented, see gradatim/frontend/translate_to_brevitas.py). "
              f"STOP.\n\tDetails: {fp_layer} of model {self.model_name}")
        exit(1)

    def translate_Linear(self, fp_layer, q_module):
        # if '2d' in self._last_layer_name:
        if len(self._last_layer_shape) > 2:
            q_module.append(Reshape(lambda x: (x.shape[0], -1)))
        in_features = fp_layer.weight.shape[1]
        out_features = fp_layer.weight.shape[0]
        bias = False
        bias_quant = None
        if hasattr(fp_layer, 'bias'):
            bias = True
            bias_quant = self._b_quant[self._bias_cnt]
            self._bias_cnt += 1
        q_module.append(qnn.QuantLinear(in_features, out_features, bias=bias, return_quant_tensor=False,
                                        weight_quant=self._w_quant[self._weight_cnt], bias_quant=bias_quant))
        self._weight_cnt += 1
        if fp_layer != self._forward_layers[-1]:
            q_module.append(nn.BatchNorm1d(out_features))
        return True

    def translate_ReLU(self, fp_layer, q_module):
        if self._do_quantization:
            q_module.append(qnn.QuantReLU(return_quant_tensor=self._return_qt, bit_width=self._bit_width))
        else:
            q_module.append(qnn.QuantReLU(act_quant=None))
        return False

    def translate_Conv2d(self, fp_layer, q_module):
        in_features = fp_layer.weight.shape[1]
        out_features = fp_layer.weight.shape[0]
        kernel_size = fp_layer.weight.shape[2]
        stride = fp_layer.stride
        padding = fp_layer.padding
        dilation = fp_layer.dilation
        groups = fp_layer.groups
        bias = False
        bias_quant = None
        if hasattr(fp_layer, 'bias'):
            bias = True
            bias_quant = self._b_quant[self._bias_cnt]
            self._bias_cnt += 1
        q_module.append(qnn.QuantConv2d(in_features, out_features, kernel_size, stride=stride, padding=padding,
                                        dilation=dilation, groups=groups,
                                        bias=bias, return_quant_tensor=False,
                                        weight_quant=self._w_quant[self._weight_cnt], bias_quant=bias_quant))
        self._weight_cnt += 1
        if fp_layer != self._forward_layers[-1]:
            q_module.append(nn.BatchNorm2d(out_features))
        return True

    def translate_AvgPool2d(self, fp_layer, q_module):
        raise NotImplementedError
        # kernel_size = 42
        # if self._do_quantization:
        #     q_module.append(qnn.QuantAvgPool2d(kernel_size, bit_width=self._bit_width))
        # else:
        #     q_module.append(nn.AvgPool2d(kernel_size))
        # return False

    def translate_MaxPool2d(self, fp_layer, q_module):
        # attr_s = fp_layer.code.split('(input, ')[1]
        # dims_sl = attr_s.split('], ')
        # kernel_size = int(dims_sl[0][1:].split(',')[0])
        # stride = int(dims_sl[1][1:].split(',')[0])
        # padding = int(dims_sl[2][1:].split(',')[0])
        # dilation = int(dims_sl[3][1:].split(',')[0])
        # ceil_mode = bool(strtobool(dims_sl[4][1:].split(',')[0].lower()))
        # return_indices = bool(strtobool(dims_sl[4][1:].split(',')[1][1:].lower()))
        kernel_size = fp_layer.kernel_size
        stride = fp_layer.stride
        padding = fp_layer.padding
        dilation = fp_layer.dilation
        ceil_mode = fp_layer.ceil_mode
        return_indices = fp_layer.return_indices
        if self._do_quantization:
            q_module.append(qnn.QuantMaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation,
                                               return_indices=return_indices, ceil_mode=ceil_mode,
                                               return_quant_tensor=self._return_qt))
        else:
            q_module.append(nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation,
                                         return_indices=return_indices, ceil_mode=ceil_mode))
        return False
