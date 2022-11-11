from brevitas.quant import *
from torch import nn
import brevitas.nn as qnn

from src.models.quantized.QTFC.QTFC import QTFC
from src.models.quantized.quant_module import QuantModule
from src.quantizers import *

dropout = 0.2
in_features = 28 * 28


class QTFCFixedPoint8(QTFC):
    """
    Activations: fixed-point 8
    Weights: fixed-point 8
    Bias: int8 (naturally fixed point, because of how Int8Bias is implemented)
    """
    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFCFixedPoint8, self).__init__(hidden1, hidden2, hidden3,
                                              act_quant=Int8ActPerTensorFixedPoint,
                                              weight_quant=Int8WeightPerTensorFixedPoint,
                                              bias_quant=Int8Bias,
                                              bit_width=8)


class QTFCFixedPoint5(QTFC):
    """
    Activations: fixed-point 5
    Weights: fixed-point 5
    Bias: int5 (naturally fixed point, because of how Int5Bias is implemented)
    """
    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFCFixedPoint5, self).__init__(hidden1, hidden2, hidden3,
                                              act_quant=Int5ActPerTensorFixedPoint,
                                              weight_quant=Int5WeightPerTensorFixedPoint,
                                              bias_quant=Int5Bias,
                                              bit_width=5)


class QTFCFixedPoint4(QTFC):
    """
    Activations: fixed-point 4
    Weights: fixed-point 4
    Bias: int4 (naturally fixed point, because of how Int4Bias is implemented)
    """
    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFCFixedPoint4, self).__init__(hidden1, hidden2, hidden3,
                                              act_quant=Int4ActPerTensorFixedPoint,
                                              weight_quant=Int4WeightPerTensorFixedPoint,
                                              bias_quant=Int4Bias,
                                              bit_width=4)


class QTFCFixedPoint3(QTFC):
    """
    Activations: fixed-point 3
    Weights: fixed-point 3
    Bias: int3 (naturally fixed point, because of how Int3Bias is implemented)
    """
    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFCFixedPoint3, self).__init__(hidden1, hidden2, hidden3,
                                              act_quant=Int3ActPerTensorFixedPoint,
                                              weight_quant=Int3WeightPerTensorFixedPoint,
                                              bias_quant=Int3Bias,
                                              bit_width=3)


