from torch import nn
import brevitas.nn as qnn
from brevitas.inject.defaults import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Int8Bias
from brevitas import config

from .quant_model import QuantModel

config.IGNORE_MISSING_KEYS = True

dropout = 0.2
in_features = 28 * 28


class QTFC(QuantModel):
    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFC, self).__init__()
        self.features.append(qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True))
        self.features.append(nn.Dropout(p=dropout))

        # first layer
        self.features.append(qnn.QuantLinear(in_features, hidden1, bias=True,
                                             weight_quant=Int8WeightPerTensorFloat,
                                             bias_quant=Int8Bias,
                                             output_quant=Int8ActPerTensorFloat,
                                             return_quant_tensor=True))
        self.features.append(nn.Dropout(p=dropout))
        self.features.append(qnn.QuantReLU(return_quant_tensor=True))

        # second layer
        self.features.append(qnn.QuantLinear(hidden1, hidden2, bias=True,
                                             weight_quant=Int8WeightPerTensorFloat,
                                             bias_quant=Int8Bias,
                                             output_quant=Int8ActPerTensorFloat,
                                             return_quant_tensor=True))
        self.features.append(nn.Dropout(p=dropout))
        self.features.append(qnn.QuantReLU(return_quant_tensor=True))

        # third layer
        self.features.append(qnn.QuantLinear(hidden2, hidden3, bias=True,
                                             weight_quant=Int8WeightPerTensorFloat,
                                             bias_quant=Int8Bias,
                                             output_quant=Int8ActPerTensorFloat,
                                             return_quant_tensor=True))
        self.features.append(nn.Dropout(p=dropout))
        self.features.append(qnn.QuantReLU(return_quant_tensor=True))

        # output layer
        self.features.append(qnn.QuantLinear(hidden3, 10, bias=True,
                                             weight_quant=Int8WeightPerTensorFloat,
                                             bias_quant=Int8Bias,
                                             output_quant=None,
                                             return_quant_tensor=False))

    def forward(self, x):
        x = x.reshape((-1, in_features))
        for module in self.features:
            x = module(x)
        return x
