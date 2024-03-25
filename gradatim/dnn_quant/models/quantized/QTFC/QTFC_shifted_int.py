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
from brevitas.quant import *

from .QTFC import QTFC
from gradatim.dnn_quant.quantizers import ShiftedUint4ActPerTensorFloat, Int4WeightPerTensorFloat, Int4Bias

dropout = 0.2
in_features = 28 * 28


class QTFCShiftedQuantAct8(QTFC):
    """
    Activations: uint8, affine
    Weights: int8, symmetric
    Bias: int8, symmetric
    """

    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFCShiftedQuantAct8, self).__init__(hidden1, hidden2, hidden3,
                                                   act_quant=ShiftedUint8ActPerTensorFloat,
                                                   weight_quant=Int8WeightPerTensorFloat,
                                                   bias_quant=Int8Bias,
                                                   bit_width=8)


class QTFCShiftedQuantAct4(QTFC):
    """
    Activations: uint8, affine
    Weights: int8, symmetric
    Bias: int8, symmetric
    """

    def __init__(self, hidden1, hidden2, hidden3):
        super(QTFCShiftedQuantAct4, self).__init__(hidden1, hidden2, hidden3,
                                                   act_quant=ShiftedUint4ActPerTensorFloat,
                                                   weight_quant=Int4WeightPerTensorFloat,
                                                   bias_quant=Int4Bias,
                                                   bit_width=4)
