from torch import nn
import src.models_processing.model_iterator as mod_it


class QuantModel(nn.Module):
    """Represents a model quantized with Brevitas"""

    def __init__(self):
        super(QuantModel, self).__init__()
        self.features = nn.ModuleList()

    def load_model_state_dict(self, fp_model):
        fp_modules = mod_it.FullPrecisionModelIterator(fp_model)
        quant_modules = mod_it.QuantModelIterator(self)

        fp_layer, q_target_type = fp_modules.find_next_weight_quantizable_module()
        while fp_layer is not None:
            q_layer = quant_modules.find_next_module_of_type(q_target_type)
            q_layer.load_state_dict(fp_layer.state_dict())

            fp_layer, q_target_type = fp_modules.find_next_weight_quantizable_module()
