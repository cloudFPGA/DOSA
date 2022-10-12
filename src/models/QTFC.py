from torch import nn
import brevitas.nn as qnn
from brevitas.inject.defaults import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Int8Bias
from brevitas import config
config.IGNORE_MISSING_KEYS = True

dropout = 0.2
in_features = 28 * 28


def __find_next_quantizable_fp_layer__(generator):
    while True:
        module = next(generator, None)
        module_name = type(module).__name__
        if module_name == 'NoneType':
            return None
        if QTFC.quantized_equivalence.get(module_name) is not None:
            return module


def __find_corresponding_q_layer__(generator, target_layer):
    q_type = QTFC.quantized_equivalence.get(type(target_layer).__name__, None)
    if q_type is None:
        raise RuntimeError

    while True:
        module = next(generator, None)
        if module is None:
            return None
        if isinstance(module, q_type):
            return module


class QTFC(nn.Module):

    quantized_equivalence = {
        'Conv1d': qnn.QuantConv1d,
        'Conv2d': qnn.QuantConv2d,
        'Linear': qnn.QuantLinear
    }

    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFC, self).__init__()

        self.features = nn.ModuleList()

        self.features.append(qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True))
        self.features.append(nn.Dropout(p=dropout))

        # first layer
        self.features.append(qnn.QuantLinear(in_features, hidden1, bias=True,
                                             weight_quant=Int8WeightPerTensorFloat,
                                             input_quant=Int8ActPerTensorFloat,
                                             bias_quant=Int8Bias,
                                             output_quant=Int8ActPerTensorFloat,
                                             return_quant_tensor=True))
        self.features.append(qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True))
        self.features.append(nn.Dropout(p=dropout))
        self.features.append(qnn.QuantReLU(return_quant_tensor=True))

        # second layer
        self.features.append(qnn.QuantLinear(hidden1, hidden2, bias=True,
                                             weight_quant=Int8WeightPerTensorFloat,
                                             bias_quant=Int8Bias,
                                             output_quant=Int8ActPerTensorFloat,
                                             return_quant_tensor=True))
        self.features.append(qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True))
        self.features.append(nn.Dropout(p=dropout))
        self.features.append(qnn.QuantReLU(return_quant_tensor=True))

        # third layer
        self.features.append(qnn.QuantLinear(hidden2, hidden3, bias=True,
                                             weight_quant=Int8WeightPerTensorFloat,
                                             bias_quant=Int8Bias,
                                             output_quant=Int8ActPerTensorFloat,
                                             return_quant_tensor=True))
        self.features.append(qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True))
        self.features.append(nn.Dropout(p=dropout))
        self.features.append(qnn.QuantReLU(return_quant_tensor=True))

        # output layer
        self.features.append(qnn.QuantLinear(hidden3, 10, bias=True))

    def forward(self, x):
        x = x.reshape((-1, in_features))
        for module in self.features:
            x = module(x)
        return x

    def load_model_state_dict(self, fp_model):
        fp_generator = fp_model.modules()
        q_generator = self.modules()

        layer = __find_next_quantizable_fp_layer__(fp_generator)
        while layer is not None:
            corresponding_layer = __find_corresponding_q_layer__(q_generator, layer)
            corresponding_layer.load_state_dict(layer.state_dict())

            layer = __find_next_quantizable_fp_layer__(fp_generator)

