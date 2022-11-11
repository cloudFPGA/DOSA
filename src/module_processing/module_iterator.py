from abc import ABC, abstractmethod
import torch

from .modules_repertory import weight_layers_all, brevitas_translation_stateful_layers


class ModuleIterator(ABC):
    def __init__(self, module):
        self.module = module
        self.modules_it = None

    @abstractmethod
    def reset(self):
        pass

    def __next__(self):
        _, module = next(self.modules_it, (None, None))
        return module

    def named_next(self):
        return next(self.modules_it, (None, None))

    def force_bias_zero(self):
        """force the module's bias to be zero (useful for debugging)"""
        self.reset()
        module = next(self)
        while module is not None:
            if type(module).__name__ in weight_layers_all and module.bias is not None:
                with torch.no_grad():
                    module.bias.fill_(0.0)
            module = next(self)
        self.reset()

    def find_next_module_of_type(self, target_type, return_name=False):
        while True:
            name, module = self.named_next()
            if module is None:
                return (None, None) if return_name else None
            if isinstance(module, target_type):
                return (name, module) if return_name else module

    def find_next_weight_module(self, return_name=False):
        while True:
            name, module = self.named_next()
            if module is None:
                return (None, None) if return_name else None

            module_type_name = type(module).__name__
            if module_type_name in weight_layers_all:
                return (name, module) if return_name else module


class QuantModuleIterator(ModuleIterator):
    """Iterates over the modules of a quantized module"""

    def __init__(self, module):
        super(QuantModuleIterator, self).__init__(module)
        self.modules_it = self.module.features.named_modules()

    def reset(self):
        self.modules_it = self.module.features.named_modules()

    def next_main_module(self, return_name=False):
        name, module = self.named_next()
        while module is not None:
            if name and name.find('.') < 0:
                return (name, module) if return_name else module
            name, module = self.named_next()
        return (None, None) if return_name else None

    def find_next_act_quant_module(self, return_name=False):
        while True:
            name, module = self.named_next()
            if module is None:
                return (None, None) if return_name else None
            if hasattr(module, 'act_quant') or hasattr(module, 'input_quant') or hasattr(module, 'output_quant'):
                return (name, module) if return_name else module

    def set_cache_inference_quant_bias(self, cache_inference_quant_bias):
        self.reset()
        module = self.find_next_act_quant_module()
        while module is not None:
            module.cache_inference_quant_bias = cache_inference_quant_bias
            module = self.find_next_act_quant_module()
        self.reset()
        pass


class FullPrecisionModuleIterator(ModuleIterator):
    """Iterates over the modules of a full precision module"""

    def __init__(self, module):
        super(FullPrecisionModuleIterator, self).__init__(module)
        self.modules_it = self.module.named_modules()

    def reset(self):
        self.modules_it = self.module.named_modules()

    def find_next_stateful_quantizable_module_with_quantized_type(self, return_name=False):
        while True:
            name, module = self.named_next()
            if module is None:
                return (None, None, None) if return_name else (None, None)

            module_type_name = type(module).__name__
            corresponding_quant_type = brevitas_translation_stateful_layers.get(module_type_name, None)
            if corresponding_quant_type is not None:
                return (name, module, corresponding_quant_type) if return_name else (module, corresponding_quant_type)
