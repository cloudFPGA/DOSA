from abc import ABC, abstractmethod
import torch

from .brevitas_nn_modules_index import weight_layers_all, brevitas_translation_stateful_layers


class ModelIterator(ABC):
    def __init__(self, model):
        self.model = model
        self.modules_it = None

    @abstractmethod
    def reset(self):
        pass

    def __next__(self):
        _, module = next(self.modules_it, (None, None))
        return module

    def named_next(self):
        return next(self.modules_it, (None, None))

    def find_next_module_of_type(self, target_type):
        while True:
            module = next(self)
            if module is None:
                return None
            if isinstance(module, target_type):
                return module

    def force_bias_zero(self):
        """force the model bias to be zero (useful for debugging)"""
        self.reset()
        module = next(self)
        while module is not None:
            if type(module).__name__ in weight_layers_all:
                with torch.no_grad():
                    module.bias.fill_(0.0)
            module = next(self)
        self.reset()


class QuantModelIterator(ModelIterator):
    """Iterates over the modules of a quantized model"""

    def __init__(self, model):
        super(QuantModelIterator, self).__init__(model)
        self.modules_it = self.model.features.named_modules()

    def reset(self):
        self.modules_it = self.model.features.named_modules()

    def find_next_act_quant_module(self):
        while True:
            module = next(self)
            if module is None:
                return None
            if hasattr(module, 'act_quant') or hasattr(module, 'input_quant') or hasattr(module, 'output_quant'):
                return module


class FullPrecisionModelIterator(ModelIterator):
    """Iterates over the modules of a full precision model"""

    def __init__(self, model):
        super(FullPrecisionModelIterator, self).__init__(model)
        self.modules_it = self.model.named_modules()

    def reset(self):
        self.modules_it = self.model.named_modules()

    def find_next_stateful_quantizable_module(self):
        while True:
            module = next(self)
            if module is None:
                return None, None

            module_name = type(module).__name__
            corresponding_quant_type = brevitas_translation_stateful_layers.get(module_name, None)
            if corresponding_quant_type is not None:
                return module, corresponding_quant_type
