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

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        name, module = next(self.modules_it)
        return name, module

    def force_bias_zero(self):
        """force the module's bias to be zero (useful for debugging)"""
        for _, module in iter(self):
            if type(module).__name__ in weight_layers_all and module.bias is not None:
                with torch.no_grad():
                    module.bias.fill_(0.0)
        self.reset()

    def find_module(self, target_module, return_name=True, **kwargs):
        stop = False
        name, module = next(self.modules_it, (None, None))
        while not stop and module is not None:
            if module is None:
                self.reset()
                stop = True
            if module is target_module:
                return (name, module) if return_name else module
            name, module = next(self.modules_it, (None, None))
        return (None, None) if return_name else None

    def find_next_module_of_type(self, target_type, return_name=False, **kwargs):
        while True:
            name, module = self.__next__()
            if module is None:
                return (None, None) if return_name else None
            if isinstance(module, target_type):
                return (name, module) if return_name else module

    def modules_of_type(self, target_type, return_name=False, **kwargs):
        new_iterator = iter(self.__class__(self.module, **kwargs))

        def impl_next(iterator):
            while True:
                name, module = next(iterator)
                if isinstance(module, target_type):
                    return (name, module) if return_name else module

        return iter(GenericIterator(inner_iterator=new_iterator, implementation_next=impl_next))

    def weight_modules(self, return_name=False, **kwargs):
        inner_it = iter(self.__class__(self.module, **kwargs))

        def impl_next(it):
            while True:
                name, module = next(it)
                if type(module).__name__ in weight_layers_all:
                    return (name, module) if return_name else module

        return iter(GenericIterator(inner_iterator=inner_it, implementation_next=impl_next))


class QuantModuleIterator(ModuleIterator):
    """Iterates over the modules of a quantized module"""

    def __init__(self, module, main_module=False):
        super(QuantModuleIterator, self).__init__(module)
        self.modules_it = self.module.features.named_modules()
        self.iter_main_modules = main_module

    def reset(self):
        self.modules_it = self.module.features.named_modules()

    def __next__(self):
        if not self.iter_main_modules:
            return super(QuantModuleIterator, self).__next__()
        else:
            name, module = super(QuantModuleIterator, self).__next__()
            while module is not None:
                if name and name.find('.') < 0:
                    return name, module
                name, module = super(QuantModuleIterator, self).__next__()
            return None, None

    def find_module(self, target_module, return_name=True, main_module=False):
        self.iter_main_modules = main_module
        return super(QuantModuleIterator, self).find_module(target_module, return_name)

    def find_next_module_of_type(self, target_type, return_name=False, main_module=False):
        self.iter_main_modules = main_module
        return super(QuantModuleIterator, self).find_next_module_of_type(target_type, return_name)

    def sub_quant_modules(self, return_name=False, main_module=False):
        from gradatim.dnn_quant.models.quantized import QuantModule
        return self.modules_of_type(QuantModule, return_name=return_name, main_module=main_module)

    def act_quant_modules(self, return_name=False, main_module=False):
        inner_it = iter(self.__class__(self.module, main_module))

        def impl_next(it):
            while True:
                name, module = next(it)
                if hasattr(module, 'act_quant') or hasattr(module, 'input_quant') or hasattr(module, 'output_quant'):
                    return (name, module) if return_name else module

        return iter(GenericIterator(inner_iterator=inner_it, implementation_next=impl_next))

    def set_cache_inference_quant_bias(self, cache_inference_quant_bias):
        for module in self.act_quant_modules():
            module.cache_inference_quant_bias = cache_inference_quant_bias


class FullPrecisionModuleIterator(ModuleIterator):
    """Iterates over the submodules of a full precision module"""

    def __init__(self, module):
        super(FullPrecisionModuleIterator, self).__init__(module)
        self.modules_it = self.module.named_modules()

    def reset(self):
        self.modules_it = self.module.named_modules()

    def stateful_quantizable_modules(self, return_name=False):
        inner_it = iter(self.__class__(self.module))

        def impl_next(it):
            while True:
                name, module = next(it)

                module_fp_type = type(module).__name__
                module_quant_type = brevitas_translation_stateful_layers.get(module_fp_type, None)
                if module_quant_type is not None:
                    return (name, module, module_quant_type) if return_name else (module, module_quant_type)

        return iter(GenericIterator(inner_iterator=inner_it, implementation_next=impl_next))


class GenericIterator:
    """ Highly polyvalent iterator, for which the iteration logic is set dynamically using lambda functions """

    def __init__(self, inner_iterator, implementation_next):
        self.inner_iterator = inner_iterator
        self.impl_next = implementation_next

    def __iter__(self):
        self.inner_iterator.reset()
        return self

    def __next__(self):
        return self.impl_next(self.inner_iterator)
