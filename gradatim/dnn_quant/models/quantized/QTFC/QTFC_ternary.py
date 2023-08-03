from brevitas.quant import SignedTernaryActPerTensorConst, SignedTernaryWeightPerTensorConst

from .QTFC import QTFC
from gradatim.dnn_quant.quantizers import Int3Bias


class QTFCTernary(QTFC):
    """
    Activations: Signed ternary
    Weights: Signed ternary
    Bias: int3
    """
    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFCTernary, self).__init__(hidden1, hidden2, hidden3,
                                          act_quant=SignedTernaryActPerTensorConst,
                                          weight_quant=SignedTernaryWeightPerTensorConst,
                                          bias_quant=Int3Bias,
                                          bit_width=2)
