from brevitas.quant_tensor import QuantTensor

from .module_iterator import QuantModuleIterator
from .modules_repertory import weight_layers_all
from src.utils import Reshape, pad_left


def describe_module(module, x=None):
    value = ''
    value_not_empty = False

    # special case: QuantModule
    from src.models.quantized import QuantModule
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
    if isinstance(x, QuantTensor):
        value += ', ' if value_not_empty else ''
        value += describe_quant_output_module(module, x_output)
        value_not_empty = True

    value += ')\n'
    return value


def describe_quant_module(module, x):
    module.eval()
    value = module._get_name() + '(\n'

    it = QuantModuleIterator(module)
    name, module = it.next_main_module(return_name=True)
    while module is not None:
        module_description = '(' + name + '): '
        module_description += describe_module(module, x)
        module_description = pad_left(module_description, 4)
        value += module_description
        x = module(x)
        name, module = it.next_main_module(return_name=True)
    value += ')\n'
    return value


def describe_quant_weight_module(module):
    value = ''
    w_scale = module.quant_weight().scale
    w_zero_point = module.quant_weight().zero_point
    value += 'weight scale: {}, '.format(w_scale.item() if w_scale is not None else 'None')
    value += 'weight zero-point: {}'.format(w_zero_point.item() if w_zero_point is not None else 'None')
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
