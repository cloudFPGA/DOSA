from src.models.quantized import QTFC
from src.quantizers import *


class QTFCInt8(QTFC):
    """
    Activations: int8, symmetric
    Weights: int8, symmetric,
    Bias: int8, symmetric
    """
    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFCInt8, self).__init__(hidden1, hidden2, hidden3,
                                       act_quant=Int8ActPerTensorFloat,
                                       weight_quant=Int8WeightPerTensorFloat,
                                       bias_quant=Int8Bias)


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


class QTFCInt4(QTFC):
    """
    Activations: int5, symmetric
    Weights: int5, symmetric,
    Bias: int5, symmetric
    """
    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFCInt4, self).__init__(hidden1, hidden2, hidden3,
                                       act_quant=Int4ActPerTensorFloat,
                                       weight_quant=Int4WeightPerTensorFloat,
                                       bias_quant=Int4Bias)



