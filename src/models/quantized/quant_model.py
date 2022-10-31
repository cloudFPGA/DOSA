from abc import ABC, abstractmethod

import torch
from brevitas.quant_tensor import QuantTensor
from torch import nn
import src.model_processing.model_iterator as iterator
from src.model_processing.brevitas_nn_modules_index import weight_layers_all
from torch.utils.tensorboard import SummaryWriter


class QuantModel(nn.Module, ABC):
    """Represents a model quantized with Brevitas"""

    def __init__(self):
        super(QuantModel, self).__init__()
        self.features = nn.ModuleList()
        self.writer = SummaryWriter(log_dir='../runs')
        self.collecting_stats = False
        self.collecting_stats_per_channel = False
        self.stats = {}

    def __str__(self):
        return self.features.__str__()

    def forward(self, x):
        if self.collecting_stats:
            return self.__collect_stats_forward__(x)

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

    def collect_stats(self, data_loader, num_iterations=30, per_channel=False):
        self.eval()
        self.collecting_stats = True
        self.collecting_stats_per_channel = per_channel
        it = iterator.QuantModelIterator(self)
        it.set_cache_inference_quant_bias(True)

        self.writer.add_graph(self, next(iter(data_loader))[0])
        self.__collect_stats_activations__(data_loader, num_iterations)
        self.__collect_stats_weights_and_bias__()

        self.collecting_stats = False
        self.__write_stats__()
        self.writer.close()

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

    def __collect_stats_weights_and_bias__(self):
        it = iterator.QuantModelIterator(self)
        name, module = it.find_next_weight_module(return_name=True)
        while module is not None:
            weights = self.__prepare_weights_bias_stats_tensors__(module.quant_weight())
            bias = self.__prepare_weights_bias_stats_tensors__(module.quant_bias())

            dict_entry_name_weights = 'weights/(' + name + '): ' + type(module).__name__
            dict_entry_name_bias = 'bias/(' + name + '): ' + type(module).__name__
            self.stats[dict_entry_name_weights] = weights
            self.stats[dict_entry_name_bias] = bias

            name, module = it.find_next_weight_module(return_name=True)

    def __collect_stats_activations__(self, data_loader, num_iterations):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        count = 0
        for values, _ in data_loader:
            if count >= num_iterations:
                break
            values = values.to(device)
            self.forward(values)
            count += 1

    def __collect_stats_forward__(self, x):
        it = iterator.QuantModelIterator(self)
        name, module = it.next_main_module(return_name=True)
        while name is not None:
            x = module(x)

            dict_entry_name = 'activations/(' + name + '): ' + type(module).__name__
            x_acc = self.stats.get(dict_entry_name, torch.empty(0))
            x_stats = self.__prepare_activation_stats_tensors__(x, x_acc, per_channel=self.collecting_stats_per_channel)
            self.stats[dict_entry_name] = x_stats

            name, module = it.next_main_module(return_name=True)
        return x

    def __write_stats__(self):
        for output_name, values in self.stats.items():
            if not isinstance(values, list):
                self.writer.add_histogram(tag=output_name, values=values, global_step=0, bins='auto')
            else:
                for i in range(len(list)):
                    self.writer.add_histogram(tag=output_name, values=values, global_step=i, bins='auto')

    @staticmethod
    def __prepare_activation_stats_tensors__(x, x_acc, per_channel=False):
        x_res = x.detach()
        if isinstance(x_res, QuantTensor):
            x_res = x.value
        x_res = x_res.detach()
        if per_channel:
            x_res = torch.transpose(x_res, 0, 1)
            x_res = x_res.view(x_res.shape[0], -1)
            concat = torch.cat((x_acc, x_res), 1)

            per_channel_tensors = []
            for i in range(concat.shape[0]):
                per_channel_tensors.append(concat[i].flatten())
            return per_channel_tensors
        return torch.cat((x_acc, x_res.flatten()), 0)

    @staticmethod
    def __prepare_weights_bias_stats_tensors__(p, per_channel=False):
        p_res = p.detach()
        if isinstance(p_res, QuantTensor):
            p_res = p_res.value
        if per_channel:
            per_channel_tensors = []
            for i in range(p_res.shape[0]):
                per_channel_tensors.append(p_res[i].flatten())
            return per_channel_tensors
        return p_res.flatten()