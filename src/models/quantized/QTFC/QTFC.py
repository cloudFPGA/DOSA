from brevitas import nn as qnn
from torch import nn

from src.models.quantized.quant_module import QuantModule
from src.utils import Reshape


class QTFC(QuantModule):
    """
    Base class for quantized TFC, per default not quantized and therefore acts as a wrapper for the full precision
    pytorch model
    """
    num_quantidd = 4
    num_linear = 4
    dropout = 0.2
    in_features = 28 * 28

    def __init__(self, hidden1, hidden2, hidden3,
                 act_quant=None,
                 weight_quant=None,
                 bias_quant=None,
                 output_quant=None,
                 bit_width=None):
        super(QTFC, self).__init__()

        return_quant_tensor = False if act_quant is None else True
        quantize_relu = act_quant is not None

        if not isinstance(act_quant, list):
            act_quant = [act_quant] * QTFC.num_quantidd

        if not isinstance(weight_quant, list):
            weight_quant = [weight_quant] * QTFC.num_linear

        if not isinstance(bias_quant, list):
            bias_quant = [bias_quant] * QTFC.num_linear

        if not isinstance(output_quant, list):
            output_quant = [output_quant] * QTFC.num_linear

        # reshape layer
        self._append(Reshape((-1, QTFC.in_features)))

        self._append(qnn.QuantIdentity(act_quant=act_quant[0], return_quant_tensor=return_quant_tensor))
        self._append(nn.Dropout(p=QTFC.dropout))

        # first layer
        self._append(qnn.QuantLinear(QTFC.in_features, hidden1, bias=True, return_quant_tensor=False,
                                     weight_quant=weight_quant[0],
                                     bias_quant=bias_quant[0],
                                     output_quant=output_quant[0]))
        self._append(nn.BatchNorm1d(hidden1))
        self._append(qnn.QuantIdentity(act_quant=act_quant[1], return_quant_tensor=return_quant_tensor))
        self._append(nn.Dropout(p=QTFC.dropout))
        if quantize_relu:
            self._append(qnn.QuantReLU(return_quant_tensor=return_quant_tensor, bit_width=bit_width))
        else:
            self._append(qnn.QuantReLU(act_quant=None, return_quant_tensor=return_quant_tensor))

        # second layer
        self._append(qnn.QuantLinear(hidden1, hidden2, bias=True, return_quant_tensor=False,
                                     weight_quant=weight_quant[1],
                                     bias_quant=bias_quant[1],
                                     output_quant=output_quant[1]))
        self._append(nn.BatchNorm1d(hidden2))
        self._append(qnn.QuantIdentity(act_quant=act_quant[2], return_quant_tensor=return_quant_tensor))
        self._append(nn.Dropout(p=QTFC.dropout))
        if quantize_relu:
            self._append(qnn.QuantReLU(return_quant_tensor=return_quant_tensor, bit_width=bit_width))
        else:
            self._append(qnn.QuantReLU(act_quant=None, return_quant_tensor=return_quant_tensor))

        # third layer
        self._append(qnn.QuantLinear(hidden2, hidden3, bias=True, return_quant_tensor=False,
                                     weight_quant=weight_quant[2],
                                     bias_quant=bias_quant[2],
                                     output_quant=output_quant[2]))
        self._append(nn.BatchNorm1d(hidden3))
        self._append(qnn.QuantIdentity(act_quant=act_quant[3], return_quant_tensor=return_quant_tensor))
        self._append(nn.Dropout(p=QTFC.dropout))
        if quantize_relu:
            self._append(qnn.QuantReLU(return_quant_tensor=return_quant_tensor, bit_width=bit_width))
        else:
            self._append(qnn.QuantReLU(act_quant=None, return_quant_tensor=return_quant_tensor))

        # output layer
        self._append(qnn.QuantLinear(hidden3, 10, bias=True, return_quant_tensor=False,
                                     weight_quant=weight_quant[3],
                                     bias_quant=bias_quant[3],
                                     output_quant=output_quant[3]))

    # def forward(self, x):
    #     x = x.reshape((-1, QTFC.in_features))
    #     return QuantModule.forward(self, x)
