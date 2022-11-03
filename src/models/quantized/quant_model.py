from abc import ABC, abstractmethod

import torch
from brevitas.quant_tensor import QuantTensor
from torch import nn
import src.model_processing.model_iterator as iterator
from src.model_processing.model_statistics import ModelStatsObserver
from src.model_processing.modules_repertory import weight_layers_all


class QuantModel(nn.Module, ABC):
    """Represents a model quantized with Brevitas"""

    def __init__(self):
        super(QuantModel, self).__init__()
        self.features = nn.ModuleList()
        self.stats_observer = ModelStatsObserver(self)

    def __str__(self):
        return self.features.__str__()

    def forward(self, x):
        for module in self.features:
            x = module(x)
        return x

    @abstractmethod
    def input_shape(self):
        pass

    def load_model_state_dict(self, fp_model):
        from brevitas import config
        config.IGNORE_MISSING_KEYS = True

        fp_modules = iterator.FullPrecisionModelIterator(fp_model)
        quant_modules = iterator.QuantModelIterator(self)

        fp_layer, q_target_type = fp_modules.find_next_stateful_quantizable_module_with_quantized_type()
        while fp_layer is not None:
            q_layer = quant_modules.find_next_module_of_type(q_target_type)
            q_layer.load_state_dict(fp_layer.state_dict())

            fp_layer, q_target_type = fp_modules.find_next_stateful_quantizable_module_with_quantized_type()

    def calibrate(self):
        self.eval()
        it = iterator.QuantModelIterator(self)
        module = it.find_next_act_quant_module()
        while module is not None:
            module.train()
            module = it.find_next_act_quant_module()

    def collect_stats(self, data_loader, num_iterations=30, per_channel=False, seed=45):
        self.stats_observer.collect_stats(data_loader, num_iterations, per_channel, seed)

    def get_quant_description(self):
        it = iterator.QuantModelIterator(self)
        x = torch.randn(self.input_shape())
        self.eval()

        value = self._get_name() + '(\n'

        name, module = it.next_main_module(return_name=True)
        while module is not None:
            value += '    (' + name + '): '
            value += module._get_name() + '('

            is_weight_layer = type(module).__name__ in weight_layers_all
            if is_weight_layer and module.quant_weight() is not None:
                wscale = module.quant_weight().scale
                wzero_point = module.quant_weight().zero_point
                value += 'weight scale: {}, '.format(wscale.item() if wscale is not None else None)
                value += 'weight zero-point: {}'.format(wzero_point.item() if wzero_point is not None else None)
            x = module(x)
            if isinstance(x, QuantTensor):
                if is_weight_layer:
                    value += ', '
                value += 'output scale: {}, '.format(x.scale.item())
                value += 'output zero-point: {}'.format(x.zero_point.item())
            value += ')\n'
            name, module = it.next_main_module(return_name=True)

        value += ')'
        return value

    def calibrating(self):
        it = iterator.QuantModelIterator(self)
        module = next(it)
        while module is not None:
            if module.training:
                return True
            module = next(it)
        return False
