from brevitas import nn as qnn
from brevitas.quant import *
from torch import nn
import brevitas.nn as qnn

from src.models.quantized import QTFC

dropout = 0.2
in_features = 28 * 28


class QTFCShiftedQuantAct(QTFC):
    """
    Activations: uint8, affine
    Weights: int8, symmetric
    Bias: int8, symmetric
    """

    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFCShiftedQuantAct, self).__init__(hidden1, hidden2, hidden3,
                                                  act_quant=ShiftedUint8ActPerTensorFloat,
                                                  weight_quant=Int8WeightPerTensorFloat,
                                                  bias_quant=Int8Bias)
