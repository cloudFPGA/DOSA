from brevitas.quant import *
from torch import nn
import brevitas.nn as qnn

from src.models.quantized.quant_model import QuantModel
from src.quantizers import *

dropout = 0.2
in_features = 28 * 28


class QTFC(QuantModel):
    """
    Activations: int8, symmetric
    Weights: int8, symmetric,
    Bias: int8, symmetric
    """
    num_quantidd = 4
    num_linear = 4

    def __init__(self, hidden1, hidden2, hidden3,
                 act_quant=Int8ActPerTensorFloat,
                 weight_quant=Int8WeightPerTensorFloat,
                 bias_quant=Int8Bias,
                 output_quant=None):

        if not isinstance(act_quant, list):
            act_quant = [act_quant] * QTFC.num_quantidd

        if not isinstance(weight_quant, list):
            weight_quant = [weight_quant] * QTFC.num_linear

        if not isinstance(bias_quant, list):
            bias_quant = [bias_quant] * QTFC.num_linear

        if not isinstance(output_quant, list):
            output_quant = [output_quant] * QTFC.num_linear

        super(QTFC, self).__init__()
        self.features.append(qnn.QuantIdentity(act_quant=act_quant[0], return_quant_tensor=True))
        self.features.append(nn.Dropout(p=dropout))

        # first layer
        self.features.append(qnn.QuantLinear(in_features, hidden1, bias=True, return_quant_tensor=False,
                                             weight_quant=weight_quant[0],
                                             bias_quant=bias_quant[0],
                                             output_quant=output_quant[0]))
        self.features.append(nn.BatchNorm1d(hidden1))
        self.features.append(qnn.QuantIdentity(act_quant=act_quant[1],
                                               return_quant_tensor=True))
        self.features.append(nn.Dropout(p=dropout))
        self.features.append(qnn.QuantReLU(return_quant_tensor=True))

        # second layer
        self.features.append(qnn.QuantLinear(hidden1, hidden2, bias=True, return_quant_tensor=False,
                                             weight_quant=weight_quant[1],
                                             bias_quant=bias_quant[1],
                                             output_quant=output_quant[1]))
        self.features.append(nn.BatchNorm1d(hidden2))
        self.features.append(qnn.QuantIdentity(act_quant=act_quant[2], return_quant_tensor=True))
        self.features.append(nn.Dropout(p=dropout))
        self.features.append(qnn.QuantReLU(return_quant_tensor=True))

        # third layer
        self.features.append(qnn.QuantLinear(hidden2, hidden3, bias=True, return_quant_tensor=False,
                                             weight_quant=weight_quant[2],
                                             bias_quant=bias_quant[2],
                                             output_quant=output_quant[2]))
        self.features.append(nn.BatchNorm1d(hidden3))
        self.features.append(qnn.QuantIdentity(act_quant=act_quant[3], return_quant_tensor=True))
        self.features.append(nn.Dropout(p=dropout))
        self.features.append(qnn.QuantReLU(return_quant_tensor=True))

        # output layer
        self.features.append(qnn.QuantLinear(hidden3, 10, bias=True, return_quant_tensor=False,
                                             weight_quant=weight_quant[3],
                                             bias_quant=bias_quant[3],
                                             output_quant=output_quant[3]))

    def forward(self, x):
        x = x.reshape((-1, in_features))
        for module in self.features:
            x = module(x)
        return x

    def input_shape(self):
        return 1, 28 * 28


class QTFCInt5(QTFC):
    """
    Activations: int5, symmetric
    Weights: int5, symmetric,
    Bias: int5, symmetric
    """
    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFCInt5, self).__init__(hidden1, hidden2, hidden3,
                                       act_quant=Int5ActPerTensorFloat,
                                       weight_quant=Int5WeightPerTensorFloat,
                                       bias_quant=Int5Bias)



