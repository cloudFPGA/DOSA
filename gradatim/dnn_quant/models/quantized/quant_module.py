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
from typing import Union, Tuple

import torch
from brevitas.quant_tensor import QuantTensor
from torch import nn, Tensor
import gradatim.dnn_quant.module_processing.module_iterator as iterator
from gradatim.dnn_quant import calibrate
from gradatim.dnn_quant.module_processing import describe_module
from gradatim.dnn_quant.module_processing.module_statistics import ModuleStatsObserver


class QuantModule(nn.Module, ABC):
    """Represents a module quantized with Brevitas"""

    def __init__(self, num_act=0, num_weighted=0, num_biased=0):
        super(QuantModule, self).__init__()
        self.features = nn.ModuleList()
        self.stats_observer = ModuleStatsObserver(self)
        self._num_idd = num_act
        self._num_weighted = num_weighted
        self._num_biased = num_biased

    def __str__(self):
        return self.features.__str__()

    def forward(self, x):
        x_next = x
        while x_next is not None:
            x = x_next
            _, _, x_next = self.forward_step(x)
        return x

    @abstractmethod
    def forward_step(self, x) -> Tuple[Union[Tensor, QuantTensor], nn.Module, Union[Tensor, QuantTensor]]:
        pass

    def it(self):
        """ Creates and return a new iterator set with self as model"""
        return iterator.QuantModuleIterator(self)

    def load_state_and_calibrate(self, fp_model, data_loader, num_steps=1, seed=0):
        self.load_module_state_dict(fp_model)
        calibrate(self, data_loader, num_steps=num_steps, seed=seed)

    def load_module_state_dict(self, fp_module):
        from brevitas import config
        config.IGNORE_MISSING_KEYS = True

        fp_modules_it = iterator.FullPrecisionModuleIterator(fp_module)
        quant_modules_it = self.it()

        for fp_layer, q_target_type in fp_modules_it.stateful_quantizable_modules():
            q_layer = quant_modules_it.find_next_module_of_type(q_target_type)
            q_layer.load_state_dict(fp_layer.state_dict())

    def calibrate(self):
        self.eval()
        it = self.it()
        for module in it.act_quant_modules():
            module.train()

    def collect_stats(self, data_loader, num_iterations=30, per_channel=False, seed=0):
        self.stats_observer.collect_stats(data_loader, num_iterations, per_channel, True, seed)

    def get_quant_description(self, input_shape):
        x = torch.randn(input_shape)
        self.eval()
        self.it().set_cache_inference_quant_bias(True)
        self.forward(x)
        return describe_module(self, x)

    def calibrating(self):
        it = self.it()
        module = next(it)
        while module is not None:
            if module.training:
                return True
            module = next(it)
        return False

    def _append(self, module):
        self.features.append(module)

    def _process_quant_methods(self, act_quant=None, weight_quant=None, bias_quant=None, bit_width=None):
        return_quant_tensor = act_quant is not None
        do_quantization = act_quant is not None
        if not isinstance(act_quant, list):
            act_quant = [act_quant] * self._num_idd
        if not isinstance(weight_quant, list):
            weight_quant = [weight_quant] * self._num_weighted
        if not isinstance(bias_quant, list):
            bias_quant = [bias_quant] * self._num_biased

        return act_quant, weight_quant, bias_quant, bit_width, return_quant_tensor, do_quantization


class GenericQuantModule(QuantModule):

    dropout = 0.2  # TODO?

    def __init__(self,
                 num_act=0,
                 num_weighted=0,
                 num_biased=0,
                 # act_quant=None,
                 # weight_quant=None,
                 # bias_quant=None,
                 # bit_width=None
                 name=''):
        super(GenericQuantModule, self).__init__(num_act=num_act, num_weighted=num_weighted, num_biased=num_biased)
        self.forward_step_index = 0
        self.name = name
        self._debug_layer_list = []

    def forward(self, x):
        for module in self.features:
            x = module(x)
        # for debugging
        # for i in range(len(self.features)):
        #     module = self.features[i]
        #     x = module(x)
        return x

    def forward_step(self, x):
        if self.forward_step_index >= len(self.features):
            self.forward_step_index = 0
            return None, None, None

        module = self.features[self.forward_step_index]
        out = module(x)
        self.forward_step_index += 1
        return x, module, out

    def append(self, module):
        self.features.append(module)
        self._debug_layer_list.append(module)

