import torch
from brevitas.quant_tensor import QuantTensor
from torch.utils.tensorboard import SummaryWriter

from src.module_processing import QuantModuleIterator, modules_repertory


class ModuleStatsObserver:
    def __init__(self, module, entries_name_prefix=None):
        self.log_dir = '../runs/' + module.__class__.__name__ + '/'
        self.module = module
        self.stats = {}
        self.entries_name_prefix = entries_name_prefix if entries_name_prefix is not None else ''

    def collect_stats(self, data_loader, num_iterations=30, per_channel=False, write_to_tensorboard=True, seed=45):
        self.module.eval()
        self.__set_quant_submodule_entries_name_prefix()

        self.__collect_weights_stats(per_channel)
        self.__collect_act_bias_stats(data_loader, num_iterations, per_channel, seed)
        self.__retrieve_quant_submodules_stats()

        if write_to_tensorboard:
            self.write_stats_to_tensorboard()

    def write_stats_to_tensorboard(self):
        writer = SummaryWriter(self.log_dir)
        for entry_name, values in self.stats.items():
            if not isinstance(values, list):
                writer.add_histogram(tag=entry_name, values=values, global_step=0, bins='auto')
            else:
                for i in range(len(list)):
                    writer.add_histogram(tag=entry_name, values=values[i], global_step=i, bins='auto')
        writer.close()

    def __set_quant_submodule_entries_name_prefix(self):
        it = QuantModuleIterator(self.module)
        for name, module in it.sub_quant_modules(return_name=True, main_module=True):
            prefix = '(' + name + '): ' + type(module).__name__ + '/'
            module.stats_observer.entries_name_prefix = prefix

    def __retrieve_quant_submodules_stats(self):
        it = QuantModuleIterator(self.module)
        for module in it.sub_quant_modules(return_name=False, main_module=True):
            self.stats.update(module.stats_observer.stats)

    def __collect_weights_stats(self, per_channel):
        it = QuantModuleIterator(self.module)
        for name, module in it.weight_modules(return_name=True, main_module=True):
            weights = ModuleStatsObserver.__prepare_stats_tensor(tensor=module.quant_weight(),
                                                                 accumulation_tensor=torch.empty(0),
                                                                 per_channel=per_channel,
                                                                 has_batch=False)
            entry_name = self.__entry_name('weights', name, module)
            self.stats[entry_name] = weights

        for module in it.sub_quant_modules(return_name=False, main_module=True):
            module.stats_observer.__collect_weights_stats(per_channel)

    def __collect_act_bias_stats(self, data_loader, num_iterations, per_channel, seed):
        it = QuantModuleIterator(self.module)
        it.set_cache_inference_quant_bias(True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.module.to(device)

        count = 0
        torch.manual_seed(seed)
        for values, _ in data_loader:
            if count >= num_iterations:
                break
            values = values.to(device)
            self.__collect_act_bias_stats_single_pass(values, per_channel)
            count += 1

    def __collect_act_bias_stats_single_pass(self, x, per_channel):
        it = QuantModuleIterator(self.module)

        x_out = x
        while x_out is not None:
            x = x_out
            x_in, sub_module, x_out = self.module.forward_step(x)
            name, _ = it.find_module(sub_module)

            from src.models.quantized import QuantModule
            if name is not None:
                if isinstance(sub_module, QuantModule):
                    sub_module.stats_observer.__collect_act_bias_stats_single_pass(x_in, per_channel)
                else:
                    self.__collect_atomic_module_act_bias_stats(x_out, name, sub_module, per_channel)

    def __collect_atomic_module_act_bias_stats(self, activations, name, module, per_channel):
        # activations
        self.__collect_module_act_stats(activations, name, module, per_channel)
        # bias
        if type(module).__name__ in modules_repertory.weight_layers_all and module.bias is not None:
            self.__collect_module_bias_stats(name, module, per_channel)

    def __collect_module_act_stats(self, activations, module_name, module, per_channel):
        a_entry_name = self.__entry_name('activations', module_name, module)
        a_accumulator = self.stats.get(a_entry_name, torch.empty(0))
        activations = ModuleStatsObserver.__prepare_stats_tensor(tensor=activations,
                                                                 accumulation_tensor=a_accumulator,
                                                                 per_channel=per_channel,
                                                                 has_batch=True)
        self.stats[a_entry_name] = activations

    def __collect_module_bias_stats(self, module_name, module, per_channel):
        b_entry_name = self.__entry_name('bias', module_name, module)
        b_accumulator = self.stats.get(b_entry_name, torch.empty(0))
        bias = ModuleStatsObserver.__prepare_stats_tensor(tensor=module.quant_bias(),
                                                          accumulation_tensor=b_accumulator,
                                                          per_channel=per_channel,
                                                          has_batch=False)
        self.stats[b_entry_name] = bias

    def __entry_name(self, param_type, module_name, module):
        return param_type + '/' + self.entries_name_prefix + '(' + module_name + '): ' + type(module).__name__

    @staticmethod
    def __prepare_stats_tensor(tensor, accumulation_tensor, per_channel, has_batch=False):
        res = tensor
        if isinstance(res, QuantTensor):
            res = tensor.value
        res = res.detach()

        if per_channel:
            if has_batch:
                res = res.transpose(0, 1)
                res = res.reshape(res.shape[0], -1)
                res = torch.cat((accumulation_tensor, res), 1)
            per_channel_tensors = []
            for i in range(res.shape[0]):
                per_channel_tensors.append(res[i].flatten())
            return per_channel_tensors

        else:
            return res.flatten()
