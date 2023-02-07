from brevitas.quant import SignedBinaryActPerTensorConst
from .QTFC import QTFC
from dnn_quant.quantizers import *

mixed_acts_quants = [
    Int3ActPerTensorFloat,
    Int6ActPerTensorFloat,
    Int8ActPerTensorFloat,
    Int4ActPerTensorFloat]

mixed_weight_quants = [
    Int8WeightPerTensorFloat,
    Int6WeightPerTensorFloat,
    Int4WeightPerTensorFloat,
    Int5WeightPerTensorFloat]

mixed_bias_quants = [
    Int8Bias,
    Int6Bias,
    Int4Bias,
    Int5Bias
]


class QTFCMixed(QTFC):
    """
    QTFC model with mixed precision: different layers use different bit widths for quantization
    """
    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFCMixed, self).__init__(hidden1, hidden2, hidden3,
                                        act_quant=mixed_acts_quants,
                                        weight_quant=mixed_weight_quants,
                                        bias_quant=mixed_bias_quants,
                                        bit_width=8)
