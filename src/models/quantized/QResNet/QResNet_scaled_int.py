from .QResNet import QResNet18
from src.quantizers import *


class QResNet18Int8(QResNet18):
    """
    Activations: int8, symmetric
    Weights: int8, symmetric,
    Bias: int8, symmetric
    """

    def __init__(self):
        super(QResNet18Int8, self).__init__(act_quant=Int8ActPerTensorFloat,
                                            weight_quant=Int8WeightPerTensorFloat,
                                            bias_quant=Int8Bias,
                                            bit_width=8)


class QResNet18Int5(QResNet18):
    """
    Activations: int5, symmetric
    Weights: int5, symmetric,
    Bias: int5, symmetric
    """

    def __init__(self):
        super(QResNet18Int5, self).__init__(act_quant=Int5ActPerTensorFloat,
                                            weight_quant=Int5WeightPerTensorFloat,
                                            bias_quant=Int5Bias,
                                            bit_width=5)


class QResNet18Int4(QResNet18):
    """
    Activations: int4, symmetric
    Weights: int4, symmetric,
    Bias: int4, symmetric
    """

    def __init__(self):
        super(QResNet18Int4, self).__init__(act_quant=Int4ActPerTensorFloat,
                                            weight_quant=Int4WeightPerTensorFloat,
                                            bias_quant=Int4Bias,
                                            bit_width=4)
