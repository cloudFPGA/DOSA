#  /*******************************************************************************
#   * Copyright 2022 -- 2024 IBM Corporation
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
from brevitas.quant_tensor import QuantTensor

from .modules_repertory import weight_layers_all
from gradatim.dnn_quant.utils import Reshape, pad_left, features_descriptions_to_string


def describe_module(module, x=None) -> str:
    value = ''
    value_not_empty = False

    # special case: QuantModule
    from gradatim.dnn_quant.models.quantized import QuantModule
    if isinstance(module, QuantModule):
        value += describe_quant_module(module, x)
        return value

    # default case: any other type of module
    value += module._get_name() + '('
    x_output = None
    if x is not None:
        x_output = module(x)

    # reshape module
    if isinstance(module, Reshape):
        value += ', ' if value_not_empty else ''
        value += describe_reshape_module(module, x)
        value_not_empty = True

    # weight layer
    if type(module).__name__ in weight_layers_all and module.quant_weight() is not None:
        value += ', ' if value_not_empty else ''
        value += describe_quant_weight_module(module)
        value_not_empty = True

    # quant output
    if isinstance(x_output, QuantTensor):
        value += ', ' if value_not_empty else ''
        value += describe_quant_output_module(module, x_output)
        value_not_empty = True

    value += ')\n'
    return value


def describe_quant_module(module, x):
    module.eval()

    features_descriptions = [None] * len(module.features)

    it = module.it()
    x_out = x
    while x_out is not None:
        x = x_out
        x_in, sub_module, x_out = module.forward_step(x)
        name, _ = it.find_module(sub_module)
        if name is not None:
            sub_module_description = '(' + name + '): '
            sub_module_description += describe_module(sub_module, x_in)
            sub_module_description = pad_left(sub_module_description, 4)
            features_descriptions[int(name)] = sub_module_description

    value = module._get_name() + '(\n'
    value += features_descriptions_to_string(features_descriptions, module.features)
    value += ')\n'
    return value


def describe_quant_weight_module(module):
    value = ''
    w_scale = module.quant_weight().scale
    w_zero_point = module.quant_weight().zero_point
    value += 'weight scale: {}, '.format(w_scale.item() if w_scale is not None else 'None')
    value += 'weight zero-point: {}'.format(w_zero_point.item() if w_zero_point is not None else 'None')
    if module.quant_bias() is not None and (module.bias > 0).any():
        b_scale = module.quant_bias().scale
        b_zero_point = module.quant_bias().zero_point
        value += ', bias scale: {}, '.format(b_scale.item() if w_scale is not None else 'None')
        value += 'bias zero-point: {}'.format(b_zero_point.item() if w_zero_point is not None else 'None')
    return value


def describe_quant_output_module(module, x):
    value = ''
    x_scale = x.scale
    x_zero_point = x.zero_point
    value += 'output scale: {}, '.format(x_scale.item() if x_scale is not None else 'None')
    value += 'output zero-point: {}'.format(x_zero_point.item() if x_zero_point is not None else 'None')
    return value


def describe_reshape_module(module, x):
    return 'target shape: {}'.format(module.shape(x))
