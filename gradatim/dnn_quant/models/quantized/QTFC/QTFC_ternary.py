#  /*******************************************************************************
#   * Copyright 2022 -- 2024 IBM Corporation
#   *
#   * Licensed under the Apache License, Version 2.0 (the "License");
#   * you may not use this file except in compliance with the License.
#   * You may obtain a copy of the License at
#   *
#   *     http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS,
#   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   * See the License for the specific language governing permissions and
#   * limitations under the License.
#  *******************************************************************************/
#
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
