from abc import ABC, abstractmethod
from typing import Union

import torch
from brevitas.quant_tensor import QuantTensor
from torch import nn, Tensor
import src.module_processing.module_iterator as iterator
from src.module_processing import describe_module
from src.module_processing.module_statistics import ModuleStatsObserver


class QuantModule(nn.Module, ABC):
    """Represents a module quantized with Brevitas"""

    def __init__(self):
        super(QuantModule, self).__init__()
        self.features = nn.ModuleList()
        self.stats_observer = ModuleStatsObserver(self)

    def __str__(self):
        return self.features.__str__()

    def forward(self, x):
        x_next = x
        while x_next is not None:
            x = x_next
            _, _, x_next = self.forward_step(x)
        return x

    @abstractmethod
    def forward_step(self, x) -> tuple[Union[Tensor, QuantTensor], nn.Module, Union[Tensor, QuantTensor]]:
        pass

    def load_module_state_dict(self, fp_module):
        from brevitas import config
        config.IGNORE_MISSING_KEYS = True

        fp_modules = iterator.FullPrecisionModuleIterator(fp_module)
        quant_modules = iterator.QuantModuleIterator(self)

        fp_layer, q_target_type = fp_modules.find_next_stateful_quantizable_module_with_quantized_type()
        while fp_layer is not None:
            q_layer = quant_modules.find_next_module_of_type(q_target_type)
            q_layer.load_state_dict(fp_layer.state_dict())

            fp_layer, q_target_type = fp_modules.find_next_stateful_quantizable_module_with_quantized_type()

    def calibrate(self):
        self.eval()
        it = iterator.QuantModuleIterator(self)
        module = it.find_next_act_quant_module()
        while module is not None:
            module.train()
            module = it.find_next_act_quant_module()

    def collect_stats(self, data_loader, num_iterations=30, per_channel=False, seed=45):
        self.stats_observer.collect_stats(data_loader, num_iterations, per_channel, seed)

    def get_quant_description(self, input_shape):
        x = torch.randn(input_shape)
        self.eval()
        return describe_module(self, x)

    def calibrating(self):
        it = iterator.QuantModuleIterator(self)
        module = next(it)
        while module is not None:
            if module.training:
                return True
            module = next(it)
        return False

    def _append(self, module):
        self.features.append(module)

