from brevitas.quant import *
from torch import nn
import brevitas.nn as qnn

from src.models.quantized import QTFC
from src.models.quantized.quant_model import QuantModel
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
                                              bias_quant=Int8Bias)


class QTFCFixedPoint5(QTFC):
    """
    Activations: fixed-point 8
    Weights: fixed-point 8
    Bias: int5 (naturally fixed point, because of how Int8Bias is implemented)
    """
    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFCFixedPoint5, self).__init__(hidden1, hidden2, hidden3,
                                              act_quant=Int5ActPerTensorFixedPoint,
                                              weight_quant=Int5WeightPerTensorFixedPoint,
                                              bias_quant=Int5Bias)

