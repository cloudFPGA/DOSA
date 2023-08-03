from brevitas.quant import SignedBinaryActPerTensorConst
from .QTFC import QTFC
from gradatim.dnn_quant.quantizers import *

mixed1_acts_quants = [
    Int3ActPerTensorFloat,
    Int6ActPerTensorFloat,
    Int8ActPerTensorFloat,
    Int4ActPerTensorFloat]

mixed1_weight_quants = [
    Int8WeightPerTensorFloat,
    Int6WeightPerTensorFloat,
    Int4WeightPerTensorFloat,
    Int5WeightPerTensorFloat]

mixed1_bias_quants = [
    Int8Bias,
    Int6Bias,
    Int4Bias,
    Int5Bias
]


class QTFCMixed1(QTFC):
    """
    QTFC model with mixed precision: different layers use different bit widths for quantization
    """
    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFCMixed1, self).__init__(hidden1, hidden2, hidden3,
                                         act_quant=mixed1_acts_quants,
                                         weight_quant=mixed1_weight_quants,
                                         bias_quant=mixed1_bias_quants,
                                         bit_width=8)


mixed2_acts_quants = [
    Int8ActPerTensorFloat,
    Int3ActPerTensorFloat,
    Int4ActPerTensorFloat,
    Int6ActPerTensorFloat,]

mixed2_weight_quants = [
    Int8WeightPerTensorFloat,
    Int4WeightPerTensorFloat,
    Int6WeightPerTensorFloat,
    Int5WeightPerTensorFloat]

mixed2_bias_quants = [
    Int8Bias,
    Int6Bias,
    Int4Bias,
    Int5Bias
]


class QTFCMixed2(QTFC):
    """
    QTFC model with mixed precision: different layers use different bit widths for quantization
    """
    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFCMixed2, self).__init__(hidden1, hidden2, hidden3,
                                         act_quant=mixed2_acts_quants,
                                         weight_quant=mixed2_weight_quants,
                                         bias_quant=mixed2_bias_quants,
                                         bit_width=8)
