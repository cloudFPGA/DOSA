import torch

from .layers_repertory import quantizable_translation_weight, weight_layers_all


class ModelIterator:
    def __init__(self, model):
        self.model = model
        self.modules_it = self.model.modules()

    def reset(self):
        self.modules_it = self.model.modules()

    def find_next_module_of_type(self, target_type):
        while True:
            module = next(self.modules_it, None)
            if module is None:
                return None
            if isinstance(module, target_type):
                return module

    def force_bias_zero(self):
        """force the model bias to be zero (useful for debugging)"""
        self.reset()
        module = next(self.modules_it, None)
        while module is not None:
            if type(module).__name__ in weight_layers_all:
                with torch.no_grad():
                    module.bias.fill_(0.0)
            module = next(self.modules_it, None)
        self.reset()


class QuantModelIterator(ModelIterator):
    """Iterates over the modules of a quantized model"""

    def __init__(self, model):
        super(QuantModelIterator, self).__init__(model)


class FullPrecisionModelIterator(ModelIterator):
    """Iterates over the modules of a full precision model"""

    def __init__(self, model):
        super(FullPrecisionModelIterator, self).__init__(model)

    def find_next_weight_quantizable_module(self):
        while True:
            module = next(self.modules_it, None)
            if module is None:
                return None, None

            module_name = type(module).__name__
            corresponding_quant_type = quantizable_translation_weight.get(module_name, None)
            if corresponding_quant_type is not None:
                return module, corresponding_quant_type
