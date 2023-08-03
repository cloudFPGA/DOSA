from brevitas.quant import SignedBinaryActPerTensorConst, SignedBinaryWeightPerTensorConst

from .QTFC import QTFC
from gradatim.dnn_quant.quantizers import Int3Bias


class QTFCBinary(QTFC):
    """
    Activations: Binary
    Weights: Binary
    Bias: Binary
    """
    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFCBinary, self).__init__(hidden1, hidden2, hidden3,
                                         act_quant=SignedBinaryActPerTensorConst,
                                         weight_quant=SignedBinaryWeightPerTensorConst,
                                         bias_quant=Int3Bias,
                                         bit_width=1)
